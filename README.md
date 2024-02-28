# Speech Data Processor

Speech Data Processor (SDP) is a toolkit to make it easy to:
1. Write code to process a new dataset, minimizing the amount of boilerplate code required.
2. Share the steps for processing a speech dataset.

SDP's philosophy is to represent processing operations as 'processor' classes, which take in a path to a NeMo-style
data manifest as input (or a path to the raw data directory if you do not have a NeMo-style manifest to start with),
apply some processing to it, and then save the output manifest file.

You specify which processors you want to run using a YAML config file. Many common processing operations are provided,
and it is easy to add your own.

![SDP overview](https://github.com/NVIDIA/NeMo/releases/download/v1.17.0/sdp_overview_diagram.png)

To learn more about SDP, have a look at our [documentation](https://nvidia.github.io/NeMo-speech-data-processor/).

## Installation

SDP is officially supported for Python 3.9, but might work for other versions.

To install all required dependencies, instead of using pip, you now need to use Docker Compose. Run the following command to set up your environment:

```
docker-compose up --build
```
This command builds and starts the containers as defined in your docker-compose.yml file. Ensure you have Docker and Docker Compose installed on your system before running this command.

To focus solely on the configuration of YouTube audio downloads via the `search_terms.json` file, here's how you could concisely describe that process:



## Configuring YouTube Audio Downloads

To tailor YouTube audio downloads to your requirements, configure the `search_terms.json` file found at:

```
sdp/processors/datasets/armdata/search_terms.json
```

This JSON file allows you to specify several parameters for downloading audio from YouTube, including the channel names, search terms, desired audio format, and the number of audio files to download. Here's the structure you should follow in the JSON file:

```json
{
    "channels": [
      {
        "channel_name": "Channel Name 1",
        "search_term": "Search Term Related to Channel 1",
        "audio_format": "wav",
        "audio_count": Number of Audios to Download
      },
      {
        "channel_name": "Channel Name 2",
        "search_term": "Search Term Related to Channel 2",
        "audio_format": "wav",
        "audio_count": Number of Audios to Download
      }
      // Add more channels as needed
    ]
}
```

- **channel_name**: The name of the YouTube channel.
- **search_term**: Specific search term to find videos from the specified channel.
- **audio_format**: The format in which you want to download the audio (e.g., "wav").
- **audio_count**: The number of audio files you wish to download for each search term.

Modify this file to include your chosen channels, search terms, desired audio format (e.g., "wav"), and the quantity of audio files you aim to download per term.


## Accessing the Docker Container
After the Docker environment has been set up and the containers are running, you will need to access the Docker container to run additional commands. Follow these steps:

List all running Docker containers to find the ID of the container you need to access:
```
docker container list
```
Copy the container ID of the relevant container from the list. The container ID is usually the first column of the output.
Use the copied container ID to access the shell inside the container:
```
docker exec -it <container_id> bash
```


After setting up the Docker environment, navigate to the ```workspace/nemo_capstone``` directory within the Docker container and run the following command to start the application with a specific configuration:
```
python main.py --config-path="dataset_configs/armenian/youtube_audio/" --config-name="config.yaml"
```
This command runs the main.py script with configuration options specified for processing Armenian audio data from YouTube. Ensure you are in the correct directory within the Docker environment before executing this command.

You will need to install additional requirements if you want to run tests or build documentation.

Some SDP processors depend on the NeMo toolkit (ASR, NLP parts) and NeMo Text Processing.
Please follow NeMo installation instructions
and NeMo Text Processing installation instructions
if you need to use such processors.




## Contributing

We welcome community contributions! Please refer to the CONTRIBUTING.md for the process.