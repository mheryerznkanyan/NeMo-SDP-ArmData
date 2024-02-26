# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import glob
import os
import urllib.request
from pathlib import Path
import subprocess
import json
import pandas as pd
from sdp.logging import logger
from pathlib import Path
# from sox import Transformer


from sdp.processors.base_processor import BaseParallelProcessor,BaseProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive

# def get_armdata_youtube_channels_list():
#     """Returns url list for CORAAL dataset.

#     There are a few mistakes in the official url list that are fixed here.
#     Can be overridden by tests to select a subset of urls.
#     """
#     dataset_url = "http://lingtools.uoregon.edu/coraal/coraal_download_list.txt"
#     urls = []
#     for file_url in urllib.request.urlopen(dataset_url):
#         file_url = file_url.decode('utf-8').strip()
#         # fixing known errors in the urls
#         if file_url == 'http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2018.10.06.txt':
#             file_url = 'http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2021.07.txt'
#         if file_url == 'http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2018.10.06.txt':
#             file_url = 'http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2021.07.txt'
#         urls.append(file_url)
#     return urls


"""
Do we need to add yt-dlp in our preprocessing part ?
Do we need to specify how many videos it has to downlad?
"""

class CreateInitialManifestArmData(BaseParallelProcessor):
    """
    Processor for creating an initial dataset manifest by saving filepaths with a common extension to the field specified in output_field.

    Args:
        raw_data_dir (str): The root directory of the files to be added to the initial manifest. This processor will recursively look for files with the extension 'extension' inside this directory.
        output_field (str): The field to store the paths to the files in the dataset.
        extension (str): The field stecify extension of the files to use them in the dataset.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        raw_data_dir: str,
        output_field: str = "audio_filepath",
        extension: str = "wav",
        **kwargs,
    ):
        super().__init__(**kwargs)
        print(1)
        self.raw_data_dir = Path(raw_data_dir)
        self.output_field = output_field
        file_path = "sdp/processors/datasets/armdata/search_terms.json"
        
        with open(file_path, "r") as f:
            channels = json.load(f)

        self.channel_tuples = [(channel["search_term"], channel["audio_count"]) for channel in channels["channels"]]


    def read_manifest(self):
        channels_data = []
        for search_term, audio_count in self.channel_tuples:
            if search_term is not None:
                command = [
                    'yt-dlp',
                    f'ytsearch{audio_count}:{search_term}',
                    '--match-filter', "license = 'Creative Commons Attribution license (reuse allowed)'",
                    '--get-id',

                ]
                    # Execute the command and capture the output
                try:

                    process = subprocess.run(command, stdout=subprocess.PIPE, text=True)
                    output = process.stdout.strip()
                    # Each video ID will be on a new line, so split the output into a list of IDs
                    video_ids = output.split('\n')

                    while("" in video_ids):
                        video_ids.remove("")
                    # Construct the full YouTube page URL for each video ID
                    youtube_base_url = "https://www.youtube.com/watch?v="
                    # Append the data to the channels_data dictionary
                    logger.info("Got youtube links :", video_ids)
                    channels_data.extend(
                        [(youtube_base_url + video_id, video_id) for video_id in video_ids]
                    )
                
                except subprocess.CalledProcessError as e:
                    print(f"Error fetching URLs for {search_term}: {e}")
            else:
                continue

        # print()
        # input_files = [str(self.raw_data_dir / file) for file in \
        #                self.raw_data_dir.rglob('*.' + self.extension)]
        return channels_data
    
    def process_dataset_entry(self, data_entry):
        # exit()
        data = {self.output_field: data_entry[0],'youtube_id':data_entry[1]}
        return [DataEntry(data=data)]


