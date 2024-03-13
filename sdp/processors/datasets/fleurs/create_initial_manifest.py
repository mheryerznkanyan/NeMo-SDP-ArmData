# # Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

import concurrent.futures
import json
import os
import typing
from typing import Tuple
from urllib.parse import parse_qs, urlparse
import urllib
from pathlib import Path

import requests

import sox
from sox import Transformer
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive


class CreateInitialManifestFleurs(BaseParallelProcessor):
    """Processor to create initial manifest for the fleurs dataset.
    Dataset link: https://huggingface.co/datasets/google/fleurs
    Will download all files in parallel and create manifest file with the
    'audio_filepath' and 'text' fields
    Args:
        config (str): Which data set shoudld be processed
            - options are:
            TODO: Add all language options in the format
            "hy_am": armenian
            "ko_kr": korean
            ["all"] (for all datasets avalable)
        split (str): Which data split should be processed
            - options are:
            "test",
            "train",
            "validation"
        audio_dir (str): Path to folder where should the filed be donwloaded and extracted
    Returns:
       This processor generates an initial manifest file with the following fields::
            {
                "audio_filepath": <path to the audio file>,
                "text": <transcription>,
            }
    """

    def __init__(
        self,
        raw_data_dir: str,
        # extract_archive_dir: str,
        # resampled_audio_dir: str,
        data_split: str,
        language_id: str,
        already_extracted: bool = False,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        # self.extract_archive_dir = extract_archive_dir
        # self.resampled_audio_dir = resampled_audio_dir
        self.data_split = data_split
        self.language_id = language_id
        self.already_extracted = already_extracted
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels
        


    def prepare(self):
        """Extracting data (unless already done)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)
        tasks = []
        self.data_rows = []
        for url in self._get_fleurs_url_list(self.language_id, self.data_split):
            rows = self._fetch_data(url)
            self.data_rows.append(rows)
            for row in rows:
                file_url = row['row']['audio'][0]['src']
                file_name = '-'.join(file_url.split('/')[-5:]).split('?')[0]
                tasks.append((file_url, str(self.raw_data_dir), file_name,False))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_file, *task) for task in tasks]

            # Wait for all futures to complete and handle exceptions
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred: {e}")
        

    def read_manifest(self):
        entries = []
        for rows in self.data_rows:
            for row in rows:
                file_url = row['row']['audio'][0]['src']
                file_transcription = row['row']['transcription']
                file_name_meta = '-'.join(file_url.split('/')[-5:])
                file_name = os.path.basename(urllib.parse.urlparse(file_name_meta).path)
                file_path = os.path.join(str(self.raw_data_dir), file_name)
                final_filepath = os.path.abspath(file_path)
                entries.append((final_filepath, file_transcription))
        return entries
   

    def process_dataset_entry(self, data_entry: Tuple[str, str]):
        file_path, text = data_entry
        
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        transcript_text = text.strip()

        # audio_path = os.path.join(self.audio_path_prefix, file_path)
        output_wav_path = os.path.join(self.raw_data_dir, file_name + ".wav")

        if not os.path.exists(output_wav_path):
            tfm = Transformer()
            tfm.rate(samplerate=self.target_samplerate)
            tfm.channels(n_channels=self.target_nchannels)
            tfm.build(input_filepath=file_path, output_filepath=output_wav_path)

        data = {
            "audio_filepath": output_wav_path,
            "duration": float(sox.file_info.duration(output_wav_path)),
            "text": transcript_text,
        }
        print(data)
        return [DataEntry(data=data)]

    def _fetch_data(self, url: str) -> list[dict[str, typing.Any]]:
        try:
            # Fetching the data from the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an error if the request failed
            data = response.json()
            rows = data.get('rows', [])
            return rows

        except requests.RequestException as e:
            print(f"Error fetching data: {e}")

    
    def _get_fleurs_url_list(self, config: str, split: str, initial_offset: int = 0, max_files: int = 3050) -> list[str]:
        # URL to fetch JSON data
        json_url = "https://datasets-server.huggingface.co/splits?dataset=google%2Ffleurs"

        # Send a request to the URL and parse the JSON response
        response = requests.get(json_url)
        if response.status_code != 200:
            raise Exception("Failed to fetch data")

        data = response.json()

        # Base URL for constructing the download URLs
        base_url = "https://datasets-server.huggingface.co/rows?dataset=google%2Ffleurs"

        # Initialize variables for fetching
        offset = initial_offset
        length = 100  
        filtered_urls = []

        while True:
            download_url = f"{base_url}&config={config}&split={split}&offset={offset}&length={length}"

            # Fetch the data
            response = requests.get(download_url)
            if response.status_code != 200:
                raise Exception("Failed to fetch data")

            batch_data = response.json()

            # Check if batch_data is not empty and add download URL to the list
            if batch_data and len(batch_data) > 0:
                print(download_url)
                filtered_urls.append(download_url)
                # Stop if the number of fetched links equals or exceeds max_links
                if max_files is not None and offset >= max_files:
                    break
                offset += length 
            else:
                break

        if len(filtered_urls) == 0:
            print(f"CONFIG: {config}\n SPLIT: {split}")
            raise ValueError("No data found for the specified config and split")
        return filtered_urls




# class CreateInitialManifestFleurs(BaseProcessor):
#     """Processor to create initial manifest for the fleurs dataset.
#     Dataset link: https://huggingface.co/datasets/google/fleurs
#     Will download all files in parallel and create manifest file with the
#     'audio_filepath' and 'text' fields
#     Args:
#         config (str): Which data set shoudld be processed
#             - options are:
#             TODO: Add all language options in the format
#             "hy_am": armenian
#             "ko_kr": korean
#             ["all"] (for all datasets avalable)
#         split (str): Which data split should be processed
#             - options are:
#             "test",
#             "train",
#             "validation"
#         audio_dir (str): Path to folder where should the filed be donwloaded and extracted
#     Returns:
#        This processor generates an initial manifest file with the following fields::
#             {
#                 "audio_filepath": <path to the audio file>,
#                 "text": <transcription>,
#             }
#     """

#     def __init__(
#         self,
#         config: str,
#         split: str,
#         audio_dir: str,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.config = config
#         self.split = split
#         self.audio_dir = audio_dir

#     def process_transcrip(self, url: str, data_folder: str) -> list[dict[str, typing.Any]]:
#         entries = []

#         data_rows = fetch_data(url)
#         for row in data_rows:
#             file_url = row['row']['audio'][0]['src']
#             file_transcription = row['row']['transcription']
#             file_name_meta = '-'.join(file_url.split('/')[-5:])
#             file_name = os.path.basename(urllib.parse.urlparse(file_name_meta).path)
#             file_path = os.path.join(data_folder, file_name)
#             entry = {}
#             entry["audio_filepath"] = os.path.abspath(file_path)
#             entry["text"] = file_transcription
#             entries.append(entry)

#         return entries

#     def process_data(self, data_folder: str, manifest_file: str) -> None:
#         entries = []

#         urls = get_fleurs_url_list(self.config, self.split)
#         for url in urls:
#             result = self.process_transcrip(url, data_folder)
#             entries.extend(result)

#         with open(manifest_file, "w") as fout:
#             for m in entries:
#                 fout.write(json.dumps(m) + "\n")

#     def download_files(self, dst_folder: str) -> None:
#         """Downloading files in parallel."""

#         os.makedirs(dst_folder, exist_ok=True)
#         tasks = []
#         for url in get_fleurs_url_list(self.config, self.split):
#             data_rows = fetch_data(url)
#             for row in data_rows:
#                 file_url = row['row']['audio'][0]['src']
#                 file_name = '-'.join(file_url.split('/')[-5:])
#                 tasks.append((file_url, str(dst_folder), file_name,False))

#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             futures = [executor.submit(download_file, *task) for task in tasks]

#             # Wait for all futures to complete and handle exceptions
#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     future.result()
#                 except Exception as e:
#                     print(f"Error occurred: {e}")

#     def process(self):
#         self.download_files(self.audio_dir)
#         self.process_data(self.audio_dir, self.output_manifest_file)