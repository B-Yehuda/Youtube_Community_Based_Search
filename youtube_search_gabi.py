import logging
import pandas as pd
from datetime import datetime

import yake

from concurrent.futures import ProcessPoolExecutor

from youtube_project.youtube_scraping_functions import search_comments, process_comments_response, \
    get_videos_from_channel, get_videos_from_query

# create logger
logger = logging.getLogger(__name__)
FORMAT = "[%(levelname)s|%(filename)s:%(lineno)s:%(funcName)s()] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)


def get_keywords(conf: tuple) -> pd.DataFrame:
    """
    Description:
        A function that uses NLP to extract keywords/key phrases from a single text.

    Background:
        Yake (Yet Another Keyword Extractor) is an unsupervised approach for automatic keyword extraction using text features.
        Yake defines a set of five features capturing keyword characteristics that are heuristically combined to assign a single score to every keyword.
        The lower the score, the more significant the keyword will be.

    Methods:
        1. KeywordExtractor - a constructor, that accepts several parameters, the most important of which are:
            maxNGrams: Maximum N-grams (number of split words) a keyword should have (Default: 3).
            minNGrams: Minimum N-grams (number of split words) a keyword should have (Default: 1).
            top: Number of words to be retrieved.
            Lan: Default is “en”.
            stopwords: A list of stop words can be passed (the words to be filtered out).
        2. KeywordExtractor.extract_keywords - a function, that return a list of tuples (keyword: score).
    """

    # extract parameters from given conf (tuple)
    text = conf[0]
    n = conf[1]
    top = conf[2]
    anchor_channel_id = conf[3]

    # build a KeywordExtractor object and pass it parameters
    kw_extractor = yake.KeywordExtractor(n=n,
                                         top=top * 5  # why 5? because we want to have some redundancy
                                         )

    # create a list of tuples (keyword: score) by passing the text to the extract_keywords function
    keywords = kw_extractor.extract_keywords(text)

    # create a data frame from the keywords and scores
    df = pd.DataFrame(keywords, columns=["Keyword", "Score (the lower the better)"])
    df["anchor_channel_id"] = anchor_channel_id

    # filter out keywords with 1 word, then save only top n_keywords
    df = df.loc[df["Keyword"].apply(lambda v: len(v.split()) > 2)]  # don't do one-word searches

    return df.sort_values("Score (the lower the better)", ascending=True).head(top)


def get_comments(params):
    """
    Description:
    A function that search comments given single video.

    Parameters:
        1. youtube: YouTube API object
        2. row: video to search comments for

    Returns:
        df_video_comments
    """

    # extract "youtube" object and "row" (i.e. video) from params
    youtube = params[0]
    row = params[1]

    # initialize next page_token to None (next page to fetch data from)
    next_page = None
    # initialize empty df to store video's comments processed data in it
    df_comments_per_video = pd.DataFrame()
    while True:
        # fetch video's comments data
        search_comments_response = search_comments(youtube=youtube,
                                                   video_id=row["video_id"],
                                                   page_token=next_page,
                                                   # verbose_cache=True
                                                   )
        if search_comments_response is None:
            break
        # process video's comments data
        next_page, df_result = process_comments_response(search_comments_response=search_comments_response,
                                                         video=row)
        # add video's comments processed data to df
        df_comments_per_video = pd.concat((df_comments_per_video, df_result))
        # set condition to follow given n_comments_per_video value
        if len(df_comments_per_video) >= 100:
            break
        if not next_page:
            break

    # retrieve anchor_channel_id
    anchor_channel_id = row["anchor_channel_id"]

    # add to each row the anchor_channel_id
    df_comments_per_video["anchor_channel_id"] = anchor_channel_id

    return df_comments_per_video


def get_videos(conf: tuple) -> (pd.DataFrame, pd.DataFrame):
    """
    Description:
    A function that retrieve Channel's/Query's videos processed data.

    Parameters:
        1. youtube: YouTube API object
        2. starting_point: Channel/Query to search videos for
        3. n_videos_per_request: How many videos to fetch per request
        4. expand_video_information: Whether to search for extra information for the videos

    Returns:
        df_videos
    """

    # extract parameters from given conf (tuple)
    youtube = conf[0]
    starting_point = conf[1]
    anchor_channel_id = conf[2]
    n_videos_per_request = conf[3]
    expand_video_information = conf[4]

    # retrieve channel's/query's videos processed data
    if starting_point.startswith("http"):
        df_videos = get_videos_from_channel(youtube=youtube,
                                            channel_url=starting_point,
                                            max_results=n_videos_per_request,
                                            expand_video_information=expand_video_information,
                                            )
        # add to each video the anchor_channel_id
        anchor_channel_id = anchor_channel_id.split("/")[-1]
        df_videos["anchor_channel_id"] = anchor_channel_id

    else:
        df_videos = get_videos_from_query(youtube=youtube,
                                          query=starting_point,
                                          max_results=n_videos_per_request,
                                          expand_video_information=expand_video_information,
                                          )
        # add to each video the anchor_channel_id
        df_videos["anchor_channel_id"] = anchor_channel_id

    return anchor_channel_id, df_videos


def get_comments_in_parallel_process(executor, youtube, videos: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
    A function that retrieve video's comments processed data in a parallel process.

    Parameters:
        1. executor: ProcessPoolExecutor object that allows parallelism of code (via multiprocessing)
        2. youtube: YouTube API object
        3. videos: Dataframe of videos

    Returns:
        df_comments
    """

    # search for each video its comments (in a parallel process)
    df_comments = list(executor.map(get_comments,
                                    [(youtube, row) for _, row in videos.iterrows()]))

    return pd.concat(df_comments)


def process_text_for_nlp(df):
    text = '\n'.join(['\n'.join(df["video_title"]),
                      '\n'.join(df["video_title"]),
                      # yes, twice, to increase the weight
                      '\n'.join(df["video_description"]),
                      # '\n'.join(df_comments["comment_text"])  # comments seem to be too noisy
                      ])
    return text


def community_based_search(*,
                           youtube,
                           df_starting_point: pd.DataFrame,
                           n_recommendations: int = 10,
                           n_videos_per_request: int = 50,
                           n_comments_per_video: int = 100,
                           keyword_target_length=5,  # to be used with yake
                           n_keywords=10,
                           ) -> pd.DataFrame:
    """
    Description:
        A function that searches for a given channel - related channels.

    Background:
        By using this function, we assume that channels are related if a YouTube visitor leaves comments on two different channels.
        Channels that share many visitors are more likely to be related.

    Methodology:
        1. Start with a seed channel (anchor_channel_id) and extract its recent videos to find related channels.
        2. Look at the video titles, descriptions, and comments and identify relevant keywords using an NLP model.
        3. Use these keywords to search for related videos on YouTube.
        4. Record which users left comments on each video and build connections between the channels.
        5. Aggregate the links (connections) and find the most related channels.

    Parameters:
        1. youtube: YouTube API object
        2. starting_point: The query to search for.
            If it's a URL, we will treat it as a channel address and extract `n_videos_per_request` most recent videos from it (i.e. PART 1)
            If it's a string, we will perform a search and extract `n_videos_per_request` most relevant videos (i.e. PART 3)
        3. n_recommendations: The number of recommendations to return
        4. n_videos_per_request: How many videos to fetch per request
        5. n_comments_per_video: How many comments to fetch per video
        6. n_keywords: Number of keywords to extract for every channel using NLP model

    Notes:
        1. Currently, we don't use the content of the videos, titles, and comments except for the initial video search.
        2. We don't use other signals such as the number of likes, view data, etc. (most of this information is available via YouTube API and can be used in the future).
        3. The resulting list is a list of creators that share audiences. Depending on the application, shared audiences might be a desired feature or a problem.
    """

    with ProcessPoolExecutor(max_workers=20) as executor:
        # PART 1 - SEARCH VIDEOS AND COMMENTS OF GIVEN CHANNEL #
        """ Search videos and comments for given channels (anchor_channel_id) requested by the user (i.e. csv input). """

        # search videos for all channels (in a parallel process)
        df_videos_dict = dict(list(executor.map(get_videos,
                                                [(youtube, row["URL"], row["URL"], n_videos_per_request, True)
                                                 for _, row in df_starting_point.iterrows()
                                                 ]
                                                )
                                   )
                              )
        df_videos = pd.concat(df_videos_dict.values())

        # search for each video its comments (in a parallel process)
        df_comments = get_comments_in_parallel_process(executor, youtube, df_videos)

        # PART 2 - USE NLP TO EXTRACT MAIN KEYWORDS FROM TEXT #
        """ Create keywords df from videos text data (comments are too noisy). """

        print(
            f"NLP keywords extraction process for all channels - started at: \033[1m{datetime.now()}\033[0m")

        # extract keywords and scores from videos text data
        df_top_keywords = pd.concat(list(executor.map(get_keywords,
                                                      [(
                                                          process_text_for_nlp(df=df_videos_of_single_channel),
                                                          5,
                                                          n_keywords,
                                                          anchor_channel_id
                                                      )
                                                          for anchor_channel_id, df_videos_of_single_channel in
                                                          df_videos_dict.items()
                                                      ]
                                                      )
                                         )
                                    )

        print(
            f"NLP keywords extraction process for all channels - finished at: \033[1m{datetime.now()}\033[0m")

        # PART 3 - SEARCH MORE VIDEOS WITH SIMILAR KEYWORDS #
        """ Expand the search to the keywords (we rely on YouTube to give us "relevant" videos). """

        print(f"Youtube search process for extracted keywords - started at: \033[1m{datetime.now()}\033[0m")

        # search for each keyword its videos (in a parallel process)
        df_more_videos = pd.concat([videos for _, videos in
                                    executor.map(get_videos,
                                                 [(youtube, row["Keyword"],
                                                   row["anchor_channel_id"],
                                                   n_videos_per_request,
                                                   False
                                                   )
                                                  for _, row in df_top_keywords.iterrows()
                                                  ]
                                                 )
                                    ]
                                   )

        # search for each video its comments (in a parallel process)
        df_more_comments = get_comments_in_parallel_process(executor, youtube, df_more_videos)

        print(f"Youtube search process for extracted keywords - finished at: \033[1m{datetime.now()}\033[0m")

        # PART 4 - UNION RESULTS #
        """ Union channel's videos (and their comments) with keywords' videos (and their comments). """

        # union dfs
        df_all_videos = pd.concat((df_videos, df_more_videos))
        df_all_comments = pd.concat((df_comments, df_more_comments)).dropna(
            # remove anonymous comments (drop rows where one of the subset columns contains null)
            subset=["comment_author_channel_id", "comment_author_channel_url"]
        )

        # get channel id and video id from df_all_videos
        channel_from_video = (
            df_all_videos[["video_channelId", "video_id"]]
            .drop_duplicates()
            .set_index("video_id")["video_channelId"]
        )

        # add to df_all_comments channel id column
        df_all_comments["video_channel_id"] = channel_from_video.reindex(df_all_comments["video_id"].values).values
        """
        Process: 
            By reindexing channel_from_video df based on df_all_comments.
            Since df_all_comments contains multiple rows for the same video_id, 
             by reindexing channel_from_video with df_all_comments["video_id"].values - 
             we get multiple video_channel_id (channel_from_video.values) for every row matching df_all_comments.
        """

        # PART 5 - COMMUNITY BASED SEARCH #
        """ Find for every channel the most related channels (by shared commenters). """

        # initialize empty df to store candidates data (candidates = related channels to given channel)
        df_candidates = pd.DataFrame()

        # anchor the main given channel (the one for which we want to look for similar channels)
        for anchor_channel_id in df_videos_dict.keys():
            print(
                f"Community based search for channel {anchor_channel_id} - started at: \033[1m{datetime.now()}\033[0m")

            # build social graph edges (where edge = channel and his commenter (i.e. video_channel_id, comment_author_channel_id))
            df_channel_edges = df_all_comments[['video_channel_id', 'comment_author_channel_id']][
                df_all_comments["anchor_channel_id"] == anchor_channel_id]

            # count number of interactions between each channel and his commenter (so more active commenters will get a higher score)
            df_channel_edges = df_channel_edges.groupby(list(df_channel_edges.columns)).agg(len).reset_index(name='n')

            # friend-of-a-friend search
            # join different channels based on similar commenters
            df_fof = df_channel_edges.merge(df_channel_edges, how='outer',
                                            on='comment_author_channel_id').drop_duplicates()

            # filter out rows where it's the same channel id
            df_fof = df_fof[df_fof.video_channel_id_x != df_fof.video_channel_id_y]

            # extract channel, related channel (by commenter), and commenter's score (by number of interactions)
            # for example: [X, Y, 14], [Y, X, 14], [X, Y, 8], [Y, X, 8]
            # TODO: Consider changing FROM score=row["n_x"]+row["n_y"] TO score=row["n_x"]*row["n_y"]
            edge_counts = df_fof.apply(lambda row: (frozenset((row["video_channel_id_x"], row["video_channel_id_y"])),
                                                    row["n_x"] + row["n_y"]
                                                    ),
                                       axis='columns'
                                       )

            # transform the extraction to list
            edge_counts = [(s, n) for (s, n) in edge_counts]

            # transform the extraction to df
            df_edge_counts = pd.DataFrame(edge_counts, columns=['Connection', 'Score'])

            # sum the score (= total number of interactions based on commenters) between 2 channels
            # (yes, with this type of calculation - the score is multiplied by 2 since the OUTER JOIN, but it doesn't mean anything... it's for all channels...)
            df_edge_counts = df_edge_counts.groupby('Connection')['Score'].agg(sum).reset_index()

            # aggregation and sorting
            # extract only rows (Connection) which contains the main given channel (anchor_channel_id)
            df_candidates_of_single_channel = df_edge_counts.loc[
                df_edge_counts["Connection"].apply(lambda c: anchor_channel_id in c)].copy()

            # add candidate channel (= related channel to anchor_channel_id) (by popping out from every row - the anchor_channel_id)
            df_candidates_of_single_channel['candidate'] = df_candidates_of_single_channel["Connection"].apply(
                lambda s: set(s.difference({anchor_channel_id})).pop())

            # sort values by score (~= total number of interactions based on commenters)
            df_candidates_of_single_channel = df_candidates_of_single_channel.sort_values('Score',
                                                                                          ascending=False).reset_index(
                drop=True)

            # alter candidate channel string to url
            df_candidates_of_single_channel['Channel URL (Output)'] = 'https://www.youtube.com/channel/' + \
                                                                      df_candidates_of_single_channel['candidate']

            # add anchor_channel_id column
            df_candidates_of_single_channel[
                "Channel URL (Input)"] = f"https://www.youtube.com/channel/{anchor_channel_id}"

            # return specified n_recommendations
            df_candidates_of_single_channel = df_candidates_of_single_channel[
                ['Channel URL (Input)', 'Channel URL (Output)', 'Score']]

            # add to df_candidates the related channels of each given anchor_channel_id
            df_candidates = pd.concat((df_candidates, df_candidates_of_single_channel.head(n_recommendations)))

            print(
                f"Community based search for channel {anchor_channel_id} - finished at: \033[1m{datetime.now()}\033[0m")

        return df_candidates


def content_based_search():
    """
    Description:
        Use the content-based search to discover creators that create similar content but do not necessarily share audiences.

    Background:
        The content-based approach aims to find channels with similar content.

    Methodology:
        To do so, we use NLP to extract relevant keywords from video titles, descriptions, and comments and perform a search for these keywords.
        Next, we build links between YouTube channels that have similar keywords in their videos.
        Channels that share more links are considered more related.

    Notes:
        NOT IMPLEMENTED YET
    """
    pass
