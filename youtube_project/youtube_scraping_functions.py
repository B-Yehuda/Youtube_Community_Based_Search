import contextlib
import json
import logging
import os
from typing import Literal

import googleapiclient.discovery
import pandas as pd
from googleapiclient.errors import HttpError

# create logger
logger = logging.getLogger(__name__)
FORMAT = "[%(levelname)s|%(filename)s:%(lineno)s:%(funcName)s()] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)


def build_youtube(api_key=None):
    """
    Description:
        A function that construct a YouTube Resource object for interacting with an API.
    """

    # retrieve YouTube API key
    if api_key is None:
        # navigate to parent dir
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(parent_dir):
            raise ValueError("Parent directory not found")

        # navigate to this dir
        # this_dir = os.path.dirname(os.path.abspath(__file__))

        # navigate to data dir
        dir_data = os.path.join(parent_dir, "data")
        if not os.path.isdir(dir_data):
            raise ValueError("Data directory not found")

        # navigate to secrets dir
        dir_secrets = os.path.join(dir_data, "secrets")
        if not os.path.isdir(dir_secrets):
            raise ValueError("Secrets directory not found")

        # navigate to file
        fn_connection = os.path.join(dir_secrets, "youtube_config.json")
        if not os.path.isfile(fn_connection):
            raise ValueError("Json file not found")

        # load file
        with open(fn_connection) as config_file:
            config = json.load(config_file)
            api_key = config["API_KEY"]
        assert api_key is not None

    # define service name and version
    api_service_name = "youtube"
    api_version = "v3"

    # construct YouTube API object
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)

    return youtube


def search_videos(youtube,
                  channel_or_query,
                  search_type: Literal["query", "channel"],
                  max_results=50,
                  order="relevance",
                  page_token=None,
                  ):
    """
    Description:
        A function that search video/channel/playlist, in our case we search video, given channel/query (query=keywords) that are passed to channelId/q parameters respectively.

    Returns:
        A collection of search results that match the query parameters specified in the API request.
        By default, a search result set identifies matching video, channel, and playlist resources, but you can also configure queries to only retrieve a specific type of resource.

    Main parameters:
        * channelId: Indicates that the API response should only contain resources created by the channel.
        * q: Specifies the query (query=keywords) term to search for.
             Request can also use the Boolean NOT (-) and OR (|) operators to exclude videos or to find videos that are associated with one of several search terms.
             For example, to search for videos matching either "boating" or "sailing", set the q parameter value to boating|sailing.
        * order: Specifies the method that will be used to order resources in the API response (e.g. date, rating, relevance etc.)
        * pageToken: Identifies a specific page in the result set that should be returned.
             In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.
    """

    # follow YouTube API limitation
    if max_results > 50:
        raise NotImplementedError("Max results > 50 not implemented")

    # define parameters to pass to youtube.search().list() method
    what = dict(part="snippet",
                order=order,
                pageToken=page_token,
                type="video",
                )
    if search_type == "channel":
        what["channelId"] = channel_or_query
    elif search_type == "query":
        what["q"] = channel_or_query
    else:
        raise ValueError("\033[1m search_type object was not defined correctly \033[0m")

    # initialize request for videos data (for given channel)
    request = youtube.search().list(**what)
    logger.debug(f"Received videos Search Response for {channel_or_query} ")

    # fetch dictionary containing videos data
    response = request.execute()

    return response


def search_comments(youtube, video_id, page_token):
    """
    Description:
        A function that search for comments given videoId/channelId/allThreadsRelatedToChannelId etc.
        Returns a list of comment threads that match the API request parameters.
    """

    # initialize request for comments data (for given video)
    request = youtube.commentThreads().list(part="snippet",
                                            videoId=video_id,
                                            pageToken=page_token
                                            )
    logger.debug(f"Received comments Search Response for {video_id} ")

    # error messages navigation
    # (some videos are closed for comments, we get warning messages in these cases, but we don't care about them)
    try:
        # TODO: Edit description and think about using/removing this part
        #  Description: This part of code navigates the error messages to devnull()
        #  ExitStack - returns object and enter_context function (which both have __enter__ and __exit__ functions)
        #  os.devnull - a point to /dev/null in Linux (when you write to /dev/null, it will discard everything received)
        with contextlib.ExitStack() as stack:
            # Redirect stderr (standard error) to os.devnull because YouTube searches generate a lot of warnings
            devnull = stack.enter_context(open(os.devnull, "w"))
            # Redirect stderr (standard error) to devnull when calling the function
            stack.enter_context(contextlib.redirect_stderr(devnull))
            # fetch dictionary containing comments data
            response = request.execute()
    except HttpError as e:
        # for example, when comments are forbidden by the author
        logger.debug(e)
        response = None

    return response


def get_more_video_info(youtube, video_id: str):
    """
    Description:
        A function that search videos, in our case we search for extra info given video (e.g. title, description, tags, contentDetails etc.), given id (video_id).
    """

    # fetch dictionary containing extra video info data
    response = youtube.videos().list(part="snippet", id=video_id).execute()

    # retrieve video_item (= video identification details) from dictionary
    video_item = response["items"][0]

    # retrieve snippet (= video extra details) from video_item
    snippet = video_item["snippet"]

    return (snippet["title"],
            snippet["description"],
            snippet.get("tags", []),
            )


def process_videos_response(youtube,
                            search_videos_response: dict,
                            expand_video_information=False
                            ) -> (str, pd.DataFrame):
    """
    Description:
        A function that process videos search results to extract/return videos' data (e.g. title, description etc.).
    """

    # get values of "items" key from dictionary containing videos data
    items = search_videos_response["items"]

    # initialize empty list to store videos data in it
    result = []

    # fetch video data from each item in "items" values
    for item in items:
        # initialize empty dictionary to store each video data in it
        item_data = dict()
        #  store video data
        for k in ["title",
                  "publishedAt",
                  "channelId",
                  "title",
                  "description",
                  "channelTitle",
                  ]:
            item_data[f"video_{k}"] = item["snippet"][k]
        item_data["video_id"] = item["id"]["videoId"]
        if expand_video_information:
            (
                item_data["video_title"],
                item_data["video_description"],
                item_data["video_tags"],
            ) = get_more_video_info(youtube, item_data["video_id"])
        else:
            item_data["video_tags"] = ""
        # store each video data dictionary
        result.append(item_data)

    # check if there's another page to fetch data from and return it
    next_page_token = search_videos_response.get("nextPageToken")

    return next_page_token, pd.DataFrame(result)


def process_comments_response(search_comments_response, df_videos_row: pd.Series) -> (str, pd.DataFrame):
    """
    Description:
        A function that processes comments search results to extract/return comments data (e.g. likeCount, textDisplay etc.).
    """

    # check if there's another page to fetch data from and return it
    next_page_token = search_comments_response.get("nextPageToken")

    # initialize empty list to store comments data in it
    result = []

    # fetch comment data from each item in "items" values
    for i, item in enumerate(search_comments_response["items"]):
        #  store comment data
        comment = item["snippet"]["topLevelComment"]
        author = comment["snippet"]["authorDisplayName"]
        author_channel_url = comment["snippet"]["authorChannelUrl"]
        if author_channel_url:
            author_channel_id = comment["snippet"]["authorChannelId"]["value"]
        else:
            # anonymous comment
            author_channel_id = None
            author_channel_url = None
        like_count = comment["snippet"]["likeCount"]
        published_at = comment["snippet"]["publishedAt"]
        comment_text = comment["snippet"]["textDisplay"]
        video_id = df_videos_row["video_id"]
        video_title = df_videos_row["video_title"]
        # store each comment data dictionary
        result.append(
            {
                "comment_id": comment["id"],
                "video_id": video_id,
                "video_title": video_title,
                "comment_author": author,
                "comment_author_channel_url": author_channel_url,
                "comment_author_channel_id": author_channel_id,
                "comment_text": comment_text,
                "comment_like_count": like_count,
                "comment_published_at": published_at,
            }
        )
        logger.debug(f"Comment: {comment_text[:50]}... for {video_title}")

    return next_page_token, pd.DataFrame(result)


def get_videos_from_channel(youtube,
                            channel_url: str,
                            max_results=50,
                            order: Literal["relevance", "date", "rating", "viewCount"] = "date",
                            expand_video_information=False,
                            ) -> pd.DataFrame:
    """
    Description:
        A function that retrieves channel's videos (by applying search_videos function on given channel_id),
        then processing the data of the retrieved videos (by applying process_videos_response function on given search_videos_response).
    """

    # fetch channel_id from given url
    channel_id = channel_url.split("/")[-1]

    # set next_page_token to None (no more pages after current page)
    next_page_token = None

    # initialize empty df to store channel's videos data in it
    df_ret = pd.DataFrame()
    while True:
        # fetch channel's videos
        search_videos_response = search_videos(youtube=youtube,
                                               channel_or_query=channel_id,
                                               search_type="channel",
                                               max_results=max_results,
                                               order=order,
                                               page_token=next_page_token,
                                               # verbose_cache=True,
                                               )
        # process videos search results to return videos data
        next_page_token, df_curr = process_videos_response(youtube=youtube,
                                                           search_videos_response=search_videos_response,
                                                           expand_video_information=expand_video_information
                                                           )
        # add channel's videos data to df
        df_ret = pd.concat([df_ret, df_curr])

        # set condition to follow YouTube API max_results limitation
        if len(df_ret) >= max_results:
            break
        if next_page_token is None:
            break

    return df_ret


def get_videos_from_query(youtube,
                          query: str,
                          max_results=50,
                          order: Literal["relevance", "date", "rating", "viewCount"] = "relevance",
                          expand_video_information=False,
                          ) -> pd.DataFrame:
    """
    Description:
        A function that retrieves query's (query=keywords) videos (by applying search_videos function on given query),
        then processing the data of the retrieved videos (by applying process_videos_response function on given search_videos_response).
    """

    # set next_page_token to None (no more pages after current page)
    next_page = None

    # initialize empty df to store query's videos data in it
    df_videos = pd.DataFrame()
    while True:
        # fetch query's videos
        search_videos_response = search_videos(youtube=youtube,
                                               channel_or_query=query,
                                               search_type="query",
                                               max_results=max_results,
                                               page_token=next_page,
                                               order=order,
                                               # verbose_cache=True,
                                               )
        # process videos search results to return videos data
        next_page, df_result = process_videos_response(youtube=youtube,
                                                       search_videos_response=search_videos_response,
                                                       expand_video_information=expand_video_information
                                                       )
        # add query's videos data to df
        df_videos = pd.concat((df_videos, df_result))

        # set condition to follow YouTube API max_results limitation
        if len(df_videos) >= max_results:
            break
        if not next_page:
            break

    return df_videos
