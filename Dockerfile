# Use the Nemo image as the base
FROM testimage:v2

# Set the working directory inside the container
WORKDIR /workspace

# Copy the requirements file from the host to the container
COPY /requirements/main.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the script executable
