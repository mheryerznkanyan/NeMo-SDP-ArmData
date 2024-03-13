# Use the Nemo image as the base
FROM nvcr.io/nvidia/nemo:23.10

# Set the working directory inside the container
WORKDIR /workspace

# Copy the requirements file from the host to the container
COPY nemo_capstone/requirements/main.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY nemo_training_pipeline.sh /workspace/nemo_capstone/nemo_training_pipeline.sh
# Make the script executable
RUN chmod +x /workspace/nemo_capstone/nemo_training_pipeline.sh

# Set the script as the entrypoint or CMD
ENTRYPOINT ["/workspace/nemo_capstone/nemo_training_pipeline.sh"]