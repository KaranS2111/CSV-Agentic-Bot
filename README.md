# Gemini-Powered CSV Agent

This project is a powerful CSV data manipulation tool that uses Google's Gemini AI to help you work with CSV files through natural language commands.

## Features

- Load and view CSV files
- Add, edit, and remove rows and columns
- Find specific data within your CSV
- Case-insensitive column name matching
- Natural language interface for data manipulation
- Download modified CSV files

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository to your local machine

2. Navigate to the project directory:
   ```bash
   cd "c:\Users\User\Desktop\CSV-Agentic-Bot"
   ```

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up your Google API key:
   - Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/)
   - Open `main.py` and replace `"enter_google_api_key_here"` with your actual API key
   - Alternatively, set it as an environment variable named `GOOGLE_API_KEY`

## Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. The Gradio interface will launch in your default web browser

3. Upload a CSV file using the file upload button

4. Interact with your data using natural language commands in the text box

### Example Commands

- "Show me the first 10 rows of data"
- "Add a new column called 'Status' with default value 'Pending'"
- "Find rows where the name is John"
- "Set the salary to 75000 for row 3"
- "Remove rows 2, 5, and 7"
- "What columns are available in this dataset?"
- "Add a new row with name='Alice', age=28, department='Engineering'"

## Project Structure

- `main.py`: The main application file containing all the code
- `requirements.txt`: List of Python dependencies

## Troubleshooting

- If you encounter an "empty text parameter" error, try rephrasing your command
- Make sure your CSV file is properly formatted
- Check that your Google API key is valid and has access to the Gemini models

## License

This project is available for personal and educational use.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Google Gemini AI](https://ai.google.dev/)
- Interface created with [Gradio](https://www.gradio.app/)
