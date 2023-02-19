import itertools
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


def get_keywords(text, **kwargs) -> pd.DataFrame:
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

    # build a KeywordExtractor object and pass it parameters
    kw_extractor = yake.KeywordExtractor(**kwargs)

    # create a list of tuples (keyword: score) by passing the text to the extract_keywords function
    keywords = kw_extractor.extract_keywords(text)

    # create a data frame from the keywords and scores
    df = pd.DataFrame(keywords, columns=["Keyword", "Score (the lower the better)"])

    return df.sort_values("Score (the lower the better)", ascending=True)


def distributed_comments_processor(params):
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
    row = params[1][1]

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
        df_video_comments = pd.concat((df_comments_per_video, df_result))
        # set condition to follow given n_comments_per_video value
        if len(df_comments_per_video) >= 100:
            break
        if not next_page:
            break
    # union all videos and their comments processed data
    return df_comments_per_video


def search_videos_and_matching_comments(*,
                                        youtube,
                                        starting_point: str,
                                        n_videos_per_request: int = 10,
                                        n_comments_per_video: int = 10,
                                        expand_video_information: bool = True,
                                        ) -> (pd.DataFrame, pd.DataFrame):
    """
    Description:
        A function that search videos given channel/query (query=keywords) and match every video it's comments.

    Returns:
             df_videos and df_comments.
    """

    # retrieve channel's/query's videos processed data
    if starting_point.startswith("http"):
        df_videos = get_videos_from_channel(youtube=youtube,
                                            channel_url=starting_point,
                                            max_results=n_videos_per_request,
                                            expand_video_information=expand_video_information,
                                            )
    else:
        df_videos = get_videos_from_query(youtube=youtube,
                                          query=starting_point,
                                          max_results=n_videos_per_request,
                                          expand_video_information=expand_video_information,
                                          )

    # retrieve video's comments processed data (distributed process)
    with ProcessPoolExecutor(max_workers=10) as executor:
        # initialize empty df comments and of all videos
        df_comments = pd.DataFrame()
        # fetch in a parallel process for each video - its comments
        for df_comments_per_video in executor.map(distributed_comments_processor,
                                                  zip(itertools.repeat(youtube, len(df_videos)), df_videos.iterrows())
                                                  ):
            df_comments = pd.concat((df_comments, df_comments_per_video))

    return df_videos, df_comments


def distributed_keywords_processor(params):
    # extract "youtube" object and "keyword" from params
    youtube = params[0]
    row = params[1][1]

    # search videos (and their comments) containing the extracted keyword and store them in df
    df_keyword_videos, df_keyword_comments = search_videos_and_matching_comments(youtube=youtube,
                                                                                 starting_point=row["Keyword"],
                                                                                 n_videos_per_request=50,
                                                                                 n_comments_per_video=50,
                                                                                 expand_video_information=False
                                                                                 )

    return df_keyword_videos, df_keyword_comments


def community_based_search(*,
                           youtube,
                           starting_point: str,
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

    Notes:
        1. Currently, we don't use the content of the videos, titles, and comments except for the initial video search.
        2. We don't use other signals such as the number of likes, view data, etc.
            (most of this information is available via YouTube API and can be used in the future).
        3. The resulting list is a list of creators that share audiences. Depending on the application, shared audiences might be a desired feature or a problem.
    """

    # PART 1 - SEARCH VIDEOS AND COMMENTS OF GIVEN CHANNEL #
    """ Search videos and comments of given channel. """

    df_videos, df_comments = search_videos_and_matching_comments(youtube=youtube,
                                                                 starting_point=starting_point,
                                                                 n_videos_per_request=n_videos_per_request,
                                                                 n_comments_per_video=n_comments_per_video,
                                                                 expand_video_information=True,
                                                                 )

    # PART 2 - USE NLP TO EXTRACT MAIN KEYWORDS FROM TEXT #
    """ Create keywords df from videos data (comments are too noisy). """

    print(f"NLP keywords extraction process for channel {starting_point} - started at: \033[1m{datetime.now()}\033[0m")

    df_keywords = get_keywords(text='\n'.join(['\n'.join(df_videos.video_title),
                                               '\n'.join(df_videos.video_title),  # yes, twice, to increase the weight
                                               '\n'.join(df_videos.video_description),
                                               # '\n'.join(df_comments.comment_text)  # comments seem to be too noisy
                                               ]
                                              ),
                               n=5,
                               top=5 * n_keywords,  # why 5? because we want to have some redundancy
                               )

    print(f"NLP keywords extraction process for channel {starting_point} - finished at: \033[1m{datetime.now()}\033[0m")

    # filter out keywords with 1 word, then save only top n_keywords
    df_keywords = df_keywords.loc[
        df_keywords["Keyword"].apply(lambda v: len(v.split()) > 2)].head(n_keywords)  # don't do one-word searches

    # PART 3 - SEARCH MORE VIDEOS WITH SIMILAR KEYWORDS #
    """ Expand the search to the keywords (we rely on YouTube to give us "relevant" videos). """

    print(f"Youtube search process for extracted keywords - started at: \033[1m{datetime.now()}\033[0m")

    # retrieve keyword data (distributed process)
    with ProcessPoolExecutor(max_workers=10) as executor:
        # initialize empty df to store comments and videos of all keywords
        df_more_videos = pd.DataFrame()
        df_more_comments = pd.DataFrame()
        # fetch in a parallel process for each keyword - it's videos and comments
        for df_keyword_videos, df_keyword_comments in executor.map(distributed_keywords_processor,
                                                                   zip(itertools.repeat(youtube, len(df_keywords)),
                                                                       df_keywords.iterrows())
                                                                   ):
            df_more_videos = pd.concat((df_more_videos, df_keyword_videos))
            df_more_comments = pd.concat((df_more_comments, df_keyword_comments))

    print(f"Youtube search process for extracted keywords - finished at: \033[1m{datetime.now()}\033[0m")

    # PART 4 - UNION AND PROCESS RESULTS #
    """ Union channel's videos (and their comments) with keywords' videos (and their comments) and process it. """

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

    print(f"Community based search - started at: \033[1m{datetime.now()}\033[0m")

    # build social graph edges (where edge = channel and his commenter (i.e. video_channel_id, comment_author_channel_id))
    df_channel_edges = df_all_comments[['video_channel_id', 'comment_author_channel_id']]

    # count number of interactions between each channel and his commenter (so more active commenters will get a higher score)
    df_channel_edges = df_channel_edges.groupby(list(df_channel_edges.columns)).agg(len).reset_index(name='n')

    # anchor the main given channel (the one for which we want to look for similar channels)
    anchor_channel_id = starting_point.split("/")[-1]

    # friend-of-a-friend search
    # join different channels based on similar commenters
    df_fof = df_channel_edges.merge(df_channel_edges, how='outer', on='comment_author_channel_id').drop_duplicates()

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
    df_candidates = df_edge_counts.loc[df_edge_counts["Connection"].apply(lambda c: anchor_channel_id in c)].copy()

    # add candidate channel (= related channel to anchor_channel_id) (by popping out from every row - the anchor_channel_id)
    df_candidates['candidate'] = df_candidates["Connection"].apply(
        lambda s: set(s.difference({anchor_channel_id})).pop())

    # sort values by score (~= total number of interactions based on commenters)
    df_candidates = df_candidates.sort_values('Score', ascending=False).reset_index(drop=True)

    # alter candidate channel string to url
    df_candidates['Channel URL (Output)'] = 'https://www.youtube.com/channel/' + df_candidates['candidate']

    # return specified n_recommendations
    df_candidates = df_candidates[['Channel URL (Output)', 'Score']].head(n_recommendations)

    print(f"Community based search - finished at: \033[1m{datetime.now()}\033[0m")

    return df_candidates


def content_based_search():
    """
    Description:
        Use the content-based search to discover creators that create similar content but do not necessarily share audiences.

    Background:
        The content-based approach aims to find channels with similar content.

    Methodology:
        We use NLP to extract relevant keywords from video titles, descriptions, and comments, then performs a search for these keywords.
        Next, we build links between YouTube channels that have similar keywords in their videos.
        Channels that share more links are considered more related.

    Notes:
        NOT IMPLEMENTED YET
    """
    pass


def recommend(*,
              youtube,
              starting_point: str,
              n_recommendations: int = 20,
              n_videos_per_request: int = 50,
              n_comments_per_video: int = 100,
              ):
    """
    Description:
        A function that uses community_based_search function.

    Parameters:
        1. starting_point: The query to search for.
            If it's a URL, we will treat it as a channel address and extract `n_videos_per_request` most recent videos from it
            If it's a string, we will perform a search and extract `n_videos_per_request` most relevant videos
        2. n_recommendations: The number of recommendations to return
        3. n_videos_per_request: How many videos to fetch per request
        4. n_comments_per_video: How many comments to fetch per video
    """

    df_recommendations = community_based_search(youtube=youtube,
                                                starting_point=starting_point,
                                                n_recommendations=n_recommendations,
                                                n_videos_per_request=n_videos_per_request,
                                                n_comments_per_video=n_comments_per_video,
                                                )

    return df_recommendations
