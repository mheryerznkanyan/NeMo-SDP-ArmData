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