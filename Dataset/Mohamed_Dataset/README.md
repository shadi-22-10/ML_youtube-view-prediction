# YouTube AI Dataset Creator

This project fetches YouTube channel and video data to build a dataset for AI tasks, such as predicting video views from thumbnails and metadata.

## Features
- Collects a list of YouTube channels by keyword, filtered by minimum subscribers and video count.
- Fetches videos from each channel published between 3 and 6 months ago.
- Saves channel and video data to CSV files for further analysis or model training.
- Avoids duplicate channels and videos.

## Requirements
- Python 3.8+
- Google API Client (`google-api-python-client`)
- Other dependencies: `requests`, `pandas`
- A valid YouTube Data API v3 key

## Setup
1. Clone this repository or copy the files to your project directory.
2. Install dependencies:
   ```bash
   pip install google-api-python-client requests pandas
   ```
3. Set your YouTube Data API key in the scripts (`API_KEY` variable).
4. (Optional) Create a Python virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```

## Usage
Run the dataset creator script:
```bash
python dataset_creator.py
```

- The script will build a channel list (`channel_list.csv`) and fetch video data (`youtube_dataset.csv`).
- You can adjust filtering criteria (subscriber count, video count, keywords, etc.) in the script config section.

## Output Files
- `channel_list.csv`: List of channels meeting criteria.
- `youtube_dataset.csv`: Video metadata and statistics for each channel.

## Customization
- Change keywords in the script to target different channel categories.
- Adjust date range, channel/video limits, or metadata fields as needed.

## License
This project is for educational and research purposes. Please respect YouTube's API terms of service and privacy policies.

## Author
Created by Mohamed Hagali
