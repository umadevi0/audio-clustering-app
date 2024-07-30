from flask import Flask, request, redirect, url_for, render_template, jsonify, session,send_from_directory
import os
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering, OPTICS
import pandas as pd
import math
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
import logging
import time
import glob
import shutil
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
import matplotlib
import soundfile as sf
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
import librosa.display

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CLIPS_FOLDER'] = 'clips'
app.config['CLUSTERS_FOLDER'] = 'clusters'
app.config['USER_DATA_FOLDER'] = 'user_data'
app.secret_key = 'supersecretkey'  # For session management

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLIPS_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLUSTERS_FOLDER'], exist_ok=True)
os.makedirs(app.config['USER_DATA_FOLDER'], exist_ok=True)

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'cred.json' #In this repository cred.json is not included due to large file issue ,users are requested to use their credentials file here for google drive integration using google_drive_api
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

def upload_file_to_google_drive(local_file_path, drive_folder_id):
    file_metadata = {'name': os.path.basename(local_file_path), 'parents': [drive_folder_id]}
    media = MediaFileUpload(local_file_path, resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

def create_drive_folder(folder_name, parent_folder_id=None):
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]

    try:
        folder = drive_service.files().create(body=file_metadata, fields='id, parents').execute()
        folder_id = folder.get('id')
        
        # Fetch additional metadata to get full path
        folder_parents = folder.get('parents')
        if folder_parents:
            parent_id = folder_parents[0]
            parent_folder = drive_service.files().get(fileId=parent_id, fields='id, name').execute()
            parent_folder_name = parent_folder.get('name', '')
            full_path = f"{parent_folder_name}/{folder_name}"
        else:
            full_path = folder_name
        
        logging.info(f"Created Google Drive folder '{folder_name}' with ID '{folder_id}' at path: {full_path}")
        
        return folder_id
    except Exception as e:
        logging.error(f"Error creating Google Drive folder '{folder_name}': {e}")
        return None

# Function to find folder by name
def find_folder_id(folder_name):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    response = drive_service.files().list(q=query, fields='files(id)').execute()
    folders = response.get('files', [])
    if folders:
        return folders[0]['id']
    else:
        return None
    

def get_file_id_from_path(file_name, folder_id):
    try:
        
        query = f"'{folder_id}' in parents and name = '{file_name}' and mimeType != 'application/vnd.google-apps.folder'"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            logging.error(f"No file found with name {file_name} in folder {folder_id}")
            return None
        return items[0]['id']
    except Exception as e:
        logging.error(f"Error getting file ID for {file_name}: {e}")
        return None


def list_files_in_drive_folder(folder_id):
    query = f"'{folder_id}' in parents"
    results = drive_service.files().list(q=query, pageSize=100, fields="files(id, name)").execute()
    items = results.get('files', [])
    return items

def download_file_from_google_drive(file_id, local_path):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        logging.info(f"Download {int(status.progress() * 100)}%.")
    fh.close()
#function for splitting large audio files into clips each of X second(X=User Input)
def split_audio(input_file, output_folder, split_duration_ms, use_drive=False, drive_folder_id=None):
    audio = AudioSegment.from_file(input_file)
    total_splits = math.ceil(len(audio) / split_duration_ms)
    for i in range(total_splits):
        start_time = i * split_duration_ms
        end_time = (i + 1) * split_duration_ms
        if end_time > len(audio):
            end_time = len(audio)
        split_audio = audio[start_time:end_time]
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{i+1}.wav")
        split_audio.export(output_file, format="wav")
        if use_drive and drive_folder_id:
            upload_file_to_google_drive(output_file, drive_folder_id)

def extract_audio_features(file_paths):
    features = []
    for file_path in file_paths:
        try:
            y, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            features.append(mfccs_mean)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
    return np.array(features)

def organize_audio_samples(input_folder, distance_threshold, clustering_method, project_folder, use_drive=False, drive_folder_id=None):
    audio_files = glob.glob(os.path.join(input_folder, '*.wav'))
    if not audio_files:
        logging.warning(f"No audio files found in {input_folder}")
        return

    features = extract_audio_features(audio_files)
    if features.size == 0:
        logging.error("No features were extracted from the audio files.")
        return

    if clustering_method == 'agglomerative':
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    elif clustering_method == 'optics':
        clustering_model = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.1)
    else:
        logging.error("Invalid clustering method specified.")
        return

    labels = clustering_model.fit_predict(features)
    logging.info(f"{clustering_method.capitalize()} clustering found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")

    # Extract silhouette scores
    silhouette_vals = silhouette_samples(features, labels)
    
    # Open a new text file for writing silhouette scores
    silhouette_file = os.path.join(project_folder, 'silhouette_scores.txt')
    with open(silhouette_file, 'w') as file:
        for label in np.unique(labels):
            if label != -1:
                cluster_silhouette_vals = silhouette_vals[labels == label]
                file.write(f"Cluster {label} Silhouette Score: {np.mean(cluster_silhouette_vals):.2f}\n")
    
    clusters_path = os.path.join(project_folder, 'clustered_clips')
    os.makedirs(clusters_path, exist_ok=True)

    clusters = {}
    for idx, label in enumerate(labels):
        audio_file = audio_files[idx]
        basename = os.path.basename(audio_file)
        
        cluster_clip_dir = os.path.join(clusters_path, f'cluster_{label}')
        os.makedirs(cluster_clip_dir, exist_ok=True)
        
        shutil.copy(audio_file, os.path.join(cluster_clip_dir, basename))
        logging.info(f"Copied clip {audio_file} to {cluster_clip_dir}")

        if label not in clusters:
            clusters[label] = []
        clusters[label].append(audio_file)

    # Calculate the overall silhouette score for the clustering
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    if n_clusters > 1:
        filtered_features = features[labels != -1]
        filtered_labels = labels[labels != -1]
        overall_silhouette = silhouette_score(filtered_features, filtered_labels)
        logging.info(f"Overall Silhouette Score: {overall_silhouette:.2f}")
    else:
        logging.info("Not enough clusters to calculate the overall silhouette score.")
    
    logging.info(f"Silhouette scores saved to {silhouette_file}")

    if use_drive and drive_folder_id:
        upload_file_to_google_drive(silhouette_file, drive_folder_id)
        for cluster_label, audio_files in clusters.items():
            cluster_drive_folder_id = create_drive_folder(f'cluster_{cluster_label}', drive_folder_id)
            for audio_file in audio_files:
                upload_file_to_google_drive(audio_file, cluster_drive_folder_id)

    logging.info('Clustering and organizing done.')
    return clusters

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        username = request.form.get('username')
        project_name = request.form.get('project_name')
        storage_option = request.form.get('storage_option')

        if not username or not project_name or not storage_option:
            return render_template('index.html', error="Please fill out all fields.")

        user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
        project_folder = os.path.join(user_folder, project_name)

        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        if os.path.exists(project_folder):
            overwrite = request.form.get('overwrite')
            if not overwrite:
                return render_template('index.html', error="Project already exists. Choose another name or overwrite the existing one.")
        else:
            os.makedirs(project_folder)

        session['username'] = username
        session['project_name'] = project_name
        session['storage_option'] = storage_option

        if storage_option == 'google_drive':
            # Replace 'your_test_folder_name' with the name of your existing 'Test' folder
            test_folder_name = 'Test'
            test_folder_id = find_folder_id(test_folder_name)
            if test_folder_id:
                # Create the username folder within the 'Test' folder
                username_folder_id = create_drive_folder(username, parent_folder_id=test_folder_id)
                if username_folder_id:
                    session['user_drive_folder_id'] = username_folder_id
                    # Create the project folder within the username folder
                    project_drive_folder_id = create_drive_folder(project_name, parent_folder_id=username_folder_id)
                    session['project_drive_folder_id'] = project_drive_folder_id
                else:
                    logging.error(f"Failed to create username folder '{username}' within 'Test' folder.")
            else:
                logging.error(f"'Test' folder not found on Google Drive.")

        return redirect(url_for('upload_file'))
    
    return render_template('index.html')
# Function to delete file from Google Drive
def delete_file_from_drive(file_id):
    try:
        drive_service.files().delete(fileId=file_id).execute()
        logging.info(f"Deleted file with ID: {file_id}")
    except Exception as e:
        logging.error(f"Failed to delete file with ID {file_id}: {e}")


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'username' not in session or 'project_name' not in session:
        return redirect(url_for('home'))

    username = session['username']
    project_name = session['project_name']
    storage_option = session['storage_option']
    project_drive_folder_id = session.get('project_drive_folder_id')
    clips_drive_folder_id=None
    clustered_clips_drive_folder_id=None

    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    os.makedirs(project_folder, exist_ok=True)
    if storage_option == 'google_drive' and project_drive_folder_id:
        clips_drive_folder_id = create_drive_folder('clips', parent_folder_id=project_drive_folder_id)
        clustered_clips_drive_folder_id = create_drive_folder('clustered_clips', parent_folder_id=project_drive_folder_id)
        session['clips_drive_folder_id'] = clips_drive_folder_id
        session['clustered_clips_drive_folder_id'] = clustered_clips_drive_folder_id



    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(project_folder, file.filename)
            file.save(file_path)
            if storage_option == 'google_drive' and project_drive_folder_id:
                # Upload the file to Google Drive and get its file ID
                file_id = upload_file_to_google_drive(file_path, project_drive_folder_id)
                logging.info(f"Uploaded file ID: {file_id}")
            else:
                file_id = None
            split_duration_ms = float(request.form['split_duration']) * 1000
            distance_threshold = float(request.form['distance_threshold'])
            clustering_method = request.form['clustering_method']

            clips_folder = os.path.join(project_folder, 'clips')
            os.makedirs(clips_folder, exist_ok=True)

            start_time = time.time()
            split_audio(file_path, clips_folder, split_duration_ms, use_drive=(storage_option == 'google_drive'), drive_folder_id=clips_drive_folder_id)
            delete_file_from_drive(file_id)
            clusters = organize_audio_samples(clips_folder, distance_threshold, clustering_method, project_folder, use_drive=(storage_option == 'google_drive'), drive_folder_id=clustered_clips_drive_folder_id)
            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"Total execution time: {total_time} seconds")
            return redirect(url_for('list_clusters'))
    return render_template('upload.html')




def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/clusters')
def list_clusters():
    if 'username' not in session or 'project_name' not in session:
        return redirect(url_for('home'))

    username = session['username']
    project_name = session['project_name']
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    clusters_path = os.path.join(project_folder, 'clustered_clips')

    clusters = {}
    for cluster in os.listdir(clusters_path):
        cluster_dir = os.path.join(clusters_path, cluster)
        if os.path.isdir(cluster_dir):
            files = [f for f in os.listdir(cluster_dir) if f.endswith('.wav')]
            clusters[cluster] = files

    return render_template('clusters.html', clusters=clusters, username=username, project_name=project_name)
def generate_spectrogram(input_file, output_folder, use_drive=False, drive_folder_id=None):
    try:
        y, sr = librosa.load(input_file)
        D = np.abs(librosa.stft(y))**2
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        df = pd.DataFrame(S_db)

        # Save the DataFrame to a CSV file
        csv_filename = os.path.join(output_folder, f"{os.path.basename(input_file).split('.')[0]}.csv")
        df.to_csv(csv_filename, index=False)
        logging.info(f"Spectrogram data saved to {csv_filename}")
        
        # Plot and save the spectrogram as an image file
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        
        output_file = os.path.join(output_folder, f"{os.path.basename(input_file).split('.')[0]}.png")
        plt.savefig(output_file)
        plt.close()
        
        # Upload to Google Drive if required
        if use_drive and drive_folder_id:
            upload_file_to_google_drive(csv_filename, drive_folder_id)
            upload_file_to_google_drive(output_file, drive_folder_id)
        
        return output_file
    except Exception as e:
        logging.error(f"Failed to generate spectrogram for {os.path.basename(input_file)}: {e}")
        return None

# Update the generate_all_spectrograms route
@app.route('/generate_all_spectrograms')
def generate_all_spectrograms():
    if 'username' not in session or 'project_name' not in session or 'storage_option' not in session:
        return redirect(url_for('home'))

    username = session['username']
    project_name = session['project_name']
    storage_option = session['storage_option']

    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    clusters_path = os.path.join(project_folder, 'clustered_clips')
    spectrograms_path = os.path.join(project_folder, 'spectrograms')
    spectrogram_drive_folder_id=None

    clusters = [d for d in os.listdir(clusters_path) if os.path.isdir(os.path.join(clusters_path, d))]
    spectrograms = {}

    # Check if Google Drive option is selected and get folder ID
    use_drive = (storage_option == 'google_drive')
    drive_folder_id = None
    if use_drive:
        drive_folder_id = find_folder_id(project_name)
        if not drive_folder_id:
            drive_folder_id = create_drive_folder(project_name)
        
        spectrogram_drive_folder_id = create_drive_folder('spectrograms', drive_folder_id)

    for cluster in clusters:
        cluster_dir = os.path.join(clusters_path, cluster)
        files = os.listdir(cluster_dir)
        spectrograms[cluster] = []

        for file in files:
            input_file = os.path.join(cluster_dir, file)
            output_folder = spectrograms_path
            os.makedirs(output_folder, exist_ok=True)

            try:
                spectrogram_file = generate_spectrogram(input_file, output_folder, use_drive, spectrogram_drive_folder_id)
                if spectrogram_file:
                    spectrograms[cluster].append(os.path.basename(spectrogram_file))
            except Exception as e:
                logging.error(f"Failed to generate spectrogram for {file}: {e}")

    return render_template('spectrograms.html', spectrograms=spectrograms)

@app.route('/clustered_clips/<cluster>/<filename>')
def serve_cluster_file(cluster, filename):
    if 'username' not in session or 'project_name' not in session:
        return redirect(url_for('home'))

    username = session['username']
    project_name = session['project_name']
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    clusters_path = os.path.join(project_folder, 'clustered_clips', cluster)
    print(f"clusters_path: {clusters_path}")
    return send_from_directory(clusters_path, filename)

@app.route('/spectrograms/<filename>')
def serve_spectrogram(filename):
    if 'username' not in session or 'project_name' not in session:
        return redirect(url_for('home'))

    username = session['username']
    project_name = session['project_name']
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    spectrograms_path = os.path.join(project_folder, 'spectrograms')

    return send_from_directory(spectrograms_path, filename)






@app.route('/download/<path:filename>')
def download_file(filename):
    username = session.get('username')
    project_name = session.get('project_name')
    if not username or not project_name:
        return redirect(url_for('home'))

    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    return send_from_directory(project_folder, filename, as_attachment=True)





@app.route('/rename_cluster/<cluster_name>', methods=['POST'])
def rename_cluster(cluster_name):
    username = session['username']
    project_name = session['project_name']
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    # Construct the path to the clustered clips folder under the project
    clustered_clips_dir = os.path.join(project_folder, 'clustered_clips')
    new_cluster_name = request.form['new_cluster_name']
    old_cluster_path = os.path.join(clustered_clips_dir, cluster_name)
    new_cluster_path = os.path.join(clustered_clips_dir, new_cluster_name)
    try:
        os.rename(old_cluster_path, new_cluster_path)
        logging.info(f"Renamed cluster {cluster_name} to {new_cluster_name}")
        return redirect(url_for('generate_all_spectrograms'))
    except Exception as e:
        logging.error(f"Error renaming cluster {cluster_name} to {new_cluster_name}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/cluster/<cluster_name>')
def cluster(cluster_name):
    username = session['username']
    project_name = session['project_name']
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    # Construct the path to the clustered clips folder under the username
    clustered_clips_dir = os.path.join(project_folder, 'clustered_clips', cluster_name)

    # Check if the directory exists
    if not os.path.exists(clustered_clips_dir):
        logging.error(f"Cluster directory {clustered_clips_dir} does not exist.")
        return jsonify({'success': False, 'error': 'Cluster directory does not exist.'}), 404

    # List files in the cluster directory
    files = os.listdir(clustered_clips_dir)

    return render_template('cluster.html', cluster=cluster_name, files=files)





@app.route('/audio_segment/<filename>')
def audio_segment(filename):
    start_time = float(request.args.get('start', 0))
    end_time = float(request.args.get('end', 0))
    min_freq = float(request.args.get('min_freq', 0))
    max_freq = float(request.args.get('max_freq', 0))
    username = session['username']
    project_name = session['project_name']
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    clips_folder = os.path.join(project_folder, 'clips')
    input_file = os.path.join(clips_folder, filename)
    y, sr = librosa.load(input_file, sr=None)

    # Debugging: Print start and end times, and frequencies
    print(f"Start time: {start_time}, End time: {end_time}")
    print(f"Min freq: {min_freq}, Max freq: {max_freq}")

    # Trim the audio segment
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    audio_segment = y[start_sample:end_sample]

    # Save the trimmed audio segment for debugging
    trimmed_segment_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"segment_trimmed_{filename}")
    sf.write(trimmed_segment_file_path, audio_segment, sr)
    print(f"Trimmed segment saved to: {trimmed_segment_file_path}")

    # Check if the frequency range is valid, swap if necessary
    if min_freq > max_freq:
        min_freq, max_freq = max_freq, min_freq
        print(f"Swapped frequencies: Min freq: {min_freq}, Max freq: {max_freq}")

    # Apply frequency filtering
    D = librosa.stft(audio_segment)
    D_mag, D_phase = librosa.magphase(D)
    freqs = librosa.fft_frequencies(sr=sr)
    print(f"Frequency bins: {freqs.shape}")
    print(f"D_mag shape: {D_mag.shape}")
    print(f"D_phase shape: {D_phase.shape}")

    # Create the frequency mask
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    print(f"Frequency mask shape: {freq_mask.shape}")
    print(f"Frequency mask: {freq_mask}")

    # Apply the frequency mask to the magnitude
    D_mag_filtered = np.zeros_like(D_mag)
    D_mag_filtered[freq_mask, :] = D_mag[freq_mask, :]

    # Combine magnitude and phase
    D_filtered = D_mag_filtered * D_phase

    # Reconstruct the filtered audio
    audio_filtered = librosa.istft(D_filtered)

    # Save the filtered segment for debugging
    segment_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"segment_filtered_{filename}")
    sf.write(segment_file_path, audio_filtered, sr)
    print(f"Filtered segment saved to: {segment_file_path}")

    # Plot and save the spectrograms for comparison
    plot_spectrogram(D_mag, sr, f"Original Spectrogram ({filename})", f"uploads/original_spectrogram_{filename}.png")
    plot_spectrogram(D_mag_filtered, sr, f"Filtered Spectrogram ({filename})", f"uploads/filtered_spectrogram_{filename}.png")

    return send_from_directory(app.config['UPLOAD_FOLDER'], f"segment_filtered_{filename}")

def plot_spectrogram(D, sr, title, output_file):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, x_axis='time', y_axis='log', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Spectrogram saved to: {output_file}")


@app.route('/save_marked_sound', methods=['POST'])
def save_marked_sound():
    start_time = float(request.form['start'])
    end_time = float(request.form['end'])
    sound_name = request.form['name']
    filename = request.form['filename']
    username = session['username']
    project_name = session['project_name']
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    clips_folder = os.path.join(project_folder, 'clips')
    input_file = os.path.join(clips_folder, filename)
    audio = AudioSegment.from_file(input_file)
    segment = audio[start_time*1000:end_time*1000]
    
    cluster_folder = os.path.join(project_folder, 'clusters')
    os.makedirs(cluster_folder, exist_ok=True)
    subfolder_path = os.path.join(cluster_folder, sound_name)
    os.makedirs(subfolder_path, exist_ok=True)
    segment_file_path = os.path.join(subfolder_path, f"{sound_name}_{filename}")
    segment.export(segment_file_path, format="wav")
    return jsonify({'success': True})

@app.route('/delete_clip', methods=['POST'])
def delete_clip():
    data = request.get_json()
    cluster = data['cluster']
    filename = data['filename']
    username = session['username']
    project_name = session['project_name']
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    # Construct the path to the clustered clips folder under the username
    clustered_clips_dir = os.path.join(project_folder, 'clustered_clips')
    cluster_dir = os.path.join(clustered_clips_dir, cluster)
    file_path = os.path.join(cluster_dir, filename)
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted clip {filename} from cluster {cluster}")
            return jsonify({'success': True})
        else:
            logging.error(f"File {filename} not found in cluster {cluster}")
            return jsonify({'success': False, 'error': 'File not found'}), 404
    except Exception as e:
        logging.error(f"Error deleting clip {filename} from cluster {cluster}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/clips/<filename>')
def serve_clip(filename):
    username = session['username']
    project_name = session['project_name']
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], username)
    project_folder = os.path.join(user_folder, project_name)
    clips_folder = os.path.join(project_folder, 'clips')
    return send_from_directory(clips_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)
