# Use the Nemo image as the base
FROM nvcr.io/nvidia/nemo:23.10

# Set the working directory inside the container
WORKDIR /workspace/nemo_capstone

# Copy the requirements file from the host to the container
COPY requirements/main.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt