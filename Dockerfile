# Base image for Python 3.10
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app using 'streamlit run'
# Streamlit will bind to 0.0.0.0 on port 8501
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]