# Start from a lightweight official Python image (Linux-based, Python 3.11 already installed)
FROM python:3.11-slim

# Set the working directory inside the container - all following commands run from here
WORKDIR /app

# Copy only requirements.txt first (not the whole project yet) - explained below
COPY requirements.txt .

# Install all required Python packages inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your project files (api/ and models/) into the container
COPY api/ ./api/
COPY models/ ./models/

# Tell Docker this container will listen on port 8000
EXPOSE 8000

# The command that runs when the container starts - launches your FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]