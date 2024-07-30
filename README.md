# Audio Clustering and Spectrogram Analysis Web Application

## Overview

This web application enables users to upload audio files, process them into segments, and organize these segments into clusters. It provides tools for visualizing, marking, and managing audio clips through spectrograms. The application also supports user and project management and integrates with Google Drive for optional cloud storage.

## Features

### User and Project Management

* **User Accounts**: Users can create accounts with a specified username and project name, ensuring personalized directories for each project.
* **Project Overwrite Option**: Users can choose to overwrite existing projects if they share the same name.
* **Storage Options**: Users can select between local storage or Google Drive for saving project data. Google Drive integration is available only for specific users with access to the company profile.

### Audio Processing and Clustering

* **File Upload**: Upload audio files in various formats.
* **Segment Splitting**: Split audio files into segments based on a user-specified duration.
* **Clustering**: Cluster segments using distance metrics and algorithms such as Agglomerative and OPTICS clustering.
* **Spectrogram Generation**: Generate visual representations of audio segments through spectrograms.

### Spectrogram Interaction

* **Zooming**: Zoom in and out on spectrograms for detailed analysis.
* **Sound Marking**: Draw boxes on spectrograms to identify and name specific sounds for future reference.
* **Audio Playback**: Playback both the original and marked audio segments.

### File and Cluster Management

* **Rename Clusters**: Rename clusters for better organization.
* **Delete Clips**: Delete specific audio clips from a cluster.
* **Generate Spectrograms**: Generate and view spectrograms for all audio files within a project.

## Requirements

* Python 3.7+
* Flask: Web framework for the application
* Librosa: Library for audio analysis
* Soundfile: Library for reading and writing sound files
* Bootstrap: For responsive UI design
* JavaScript: For interactive features in the browser

## Setup and Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/umadevi0/audio-clustering-app.git
    cd audio-clustering-app
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Google Drive Integration (Optional only for authorized)**

    * Configure a Google Cloud project and enable the Drive API.
    * Obtain credentials and share the necessary Google Drive folder with the service account.

4. **Run the Application**

    ```bash
    python app.py
    ```

    The application will be accessible at [http://localhost:5000](http://localhost:5000).

## Usage

### Creating a Project

1. **Start a New Project**

    * Navigate to the home page.
    * Enter your username and project name.
    * Choose the storage option (Local or Google Drive).
    * If a project with the same name exists, select the overwrite option if desired.
    * Click the “Create Project” button to proceed.

2. **Upload and Process Audio Files**

    * Go to the upload page.
    * Select your audio file and specify the split duration (can be decimal or integer).
    * Set the clustering distance threshold (recommended range: 50-60) and choose the clustering method (Agglomerative or OPTICS).
    * Click the “Upload and Process” button.

    The audio file will be split into segments based on the provided duration. Segments are saved in the `clips` folder and clustered into folders under the `clustered_clips` directory. The silhouette scores of each cluster are saved in `silhouette_scores.txt`, along with the overall silhouette score and total execution time displayed in the terminal.

3. **Viewing and Managing Clusters**

    * Access the clusters page to view a list of clusters.
    * Click on a cluster name to view its detailed spectrograms.
    * Use the “Generate Spectrograms” button to create spectrograms for each audio clip. Metadata is saved in CSV files within the `spectrograms` folder.

### Spectrogram Interaction

* **Viewing Spectrograms**

    * Click on a cluster to view individual spectrograms.
    * Spectrograms include options to zoom in and out, mark specific sounds, and playback the audio.
    * To mark a sound, draw a box on the spectrogram, enter a name for the marked sound, and save it as a WAV file under the respective cluster.

* **Sound Marking and Management**

    * **Mark Sound**: Highlight specific areas on the spectrogram and name them.
    * **Unmark Sound**: Clear all highlights on the spectrogram.
    * **Delete Clip**: Remove a specific clip from the cluster. All related data (e.g., spectrograms, buttons) will be cleared.

## File Structure

* **app.py**: The primary application file that contains the routes and business logic for the web application.
* **templates/**: Directory containing HTML templates used for rendering web pages.
* **uploads/**: Directory designated for storing uploaded audio files and processed audio clips.
* **project_name/clips/**: Directory within a project that holds the segmented audio clips.
* **project_name/clustered_clips/**: Directory within a project where audio clips are organized into clusters.
* **project_name/spectrograms/**: Directory within a project dedicated to storing generated spectrograms and their associated metadata in CSV files.
* **project_name/silhouette_scores.txt**: File containing information about the silhouette scores for each cluster, used for evaluating clustering quality.
* **user_data/username/**: Directory for storing all projects associated with a particular user.
* **username/project/**: Directory for managing all files and subdirectories related to a specific project.
* **cred.json**: JSON file containing credentials required for accessing and utilizing Google Drive integration.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or support, please contact [umadevi03d@gmail.com](mailto:umadevi03d@gmail.com).

## Project Report

For a comprehensive overview of the project, including methodology, results, and future improvements, please refer to the [Project Report](https://docs.google.com/document/d/16hGEyBQRY-lbgmEja-myoUJOYsDJfsWf0twVPF_UO1o/edit?usp=sharing)
