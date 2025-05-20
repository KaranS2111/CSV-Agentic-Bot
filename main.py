import os
import pandas as pd
import gradio as gr
import google.generativeai as genai
from typing import Dict, List, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field  

# Gemini API Key Setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "enter_google_api_key_here")
genai.configure(api_key=GOOGLE_API_KEY)

# Global variables
loaded_data = None
current_file_name = None

# Tool input schemas
class RemoveRowsInput(BaseModel):
    indices: List[int] = Field(..., description="List of row indices to remove")

class RemoveColumnsInput(BaseModel):
    columns: List[str] = Field(..., description="List of column names to remove")


class AddRowInput(BaseModel):
    values: Any = Field(..., description="Dictionary of column names and values (can be a dict or a JSON string)")

class AddColumnInput(BaseModel):
    column_name: str = Field(..., description="Name of the new column")
    default_value: Any = Field(..., description="Default value")

class SetCellValueInput(BaseModel):
    row_index: Any = Field(..., description="Row index (can be int or float)")
    column_name: str = Field(..., description="Column name")
    value: Any = Field(..., description="New value")

class SetRowValuesInput(BaseModel):
    row_index: Any = Field(..., description="Row index (can be int, float, or string for name-based lookup)")
    values: Any = Field(..., description="Dictionary of column names and values (can be a dict or a JSON string)")

# Tool implementations
@tool
def load_csv(file_path: str) -> str:
    """Load a CSV file from the specified path."""
    global loaded_data, current_file_name
    try:
        loaded_data = pd.read_csv(file_path)
        current_file_name = file_path
        return f"Loaded CSV. Shape: {loaded_data.shape}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def show_data(num_rows: int = 5) -> str:
    """Display the first few rows of the loaded CSV data."""
    global loaded_data
    if loaded_data is None:
        return "No data loaded."
    return str(loaded_data.head(num_rows))

@tool
def get_data_info() -> str:
    """Get information about the loaded CSV data including row count and column names."""
    global loaded_data
    if loaded_data is None:
        return "No data loaded."
    
    row_count = len(loaded_data)
    column_count = len(loaded_data.columns)
    return f"Dataset has {row_count} rows and {column_count} columns.\nColumns: {loaded_data.columns.tolist()}"

@tool
def remove_rows(input: RemoveRowsInput) -> str:
    """Remove rows from the loaded data by their indices."""
    global loaded_data
    try:
        # Validate indices
        if loaded_data is None:
            return "No data loaded."
            
        row_count = len(loaded_data)
        invalid_indices = []
        valid_indices = []
        
        for idx in input.indices:
            actual_idx = idx if idx >= 0 else row_count + idx
            
            if actual_idx < 0 or actual_idx >= row_count:
                invalid_indices.append(idx)
            else:
                valid_indices.append(idx)
        
        if invalid_indices:
            return f"Error: Invalid row indices {invalid_indices}. Valid row indices are 0 to {row_count-1} or -{row_count} to -1. Dataset has {row_count} rows."
        
        loaded_data = loaded_data.drop(valid_indices)
        return f"Removed rows with indices {valid_indices}. New shape: {loaded_data.shape}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def remove_columns(input: RemoveColumnsInput) -> str:
    """Remove columns from the loaded data by their names."""
    global loaded_data
    try:
        missing_columns = [col for col in input.columns if col not in loaded_data.columns]
        if missing_columns:
            return f"Error: Columns {missing_columns} not found. Available columns: {loaded_data.columns.tolist()}"
            
        loaded_data = loaded_data.drop(columns=input.columns)
        return f"Removed columns: {input.columns}. New columns: {loaded_data.columns.tolist()}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def add_row(input: AddRowInput) -> str:
    """Add a new row to the loaded data with the specified values."""
    global loaded_data
    try:
        
        values = input.values
        if isinstance(values, str):
            import json
            try:
                values = json.loads(values)
            except json.JSONDecodeError as e:
                return f"Error parsing values: {str(e)}. Please provide a valid JSON object."
        
        
        if not isinstance(values, dict):
            return f"Error: values must be a dictionary or a JSON string representing a dictionary, got {type(values)}"
        
        
        if loaded_data is not None:
            normalized_values = {}
            existing_columns = loaded_data.columns.tolist()
            existing_columns_lower = [col.lower() for col in existing_columns]
            
            for key, value in values.items():
                
                if key in existing_columns:
                    normalized_values[key] = value
                
                elif key.lower() in existing_columns_lower:
                    correct_key = existing_columns[existing_columns_lower.index(key.lower())]
                    normalized_values[correct_key] = value
                else:
                    
                    normalized_values[key] = value
            
            values = normalized_values
        
        
        new_row = pd.DataFrame([values])
        
        
        loaded_data = pd.concat([new_row, loaded_data], ignore_index=True)
        
        
        return f"Added row at the top. New shape: {loaded_data.shape}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def find_row_index(column_name: str, value: str) -> str:
    """Find the index of a row where the specified column has the given value."""
    global loaded_data
    if loaded_data is None:
        return "No data loaded."
    
    try:
        
        column_map = {col.lower(): col for col in loaded_data.columns}
        col_name = column_name
        
        if col_name.lower() in column_map:
            col_name = column_map[col_name.lower()]
        else:
            return f"Column '{column_name}' not found. Available columns: {loaded_data.columns.tolist()}"
        
        
        matches = loaded_data[loaded_data[col_name].astype(str).str.lower() == value.lower()]
        
        if len(matches) == 0:
            return f"No rows found where {col_name} is '{value}'."
        
        indices = matches.index.tolist()
        return f"Found {len(indices)} row(s) with {col_name}='{value}' at indices: {indices}"
    except Exception as e:
        return f"Error: {str(e)}"


def display_csv():
    global loaded_data
    if loaded_data is None:
        return pd.DataFrame(), "No CSV loaded yet."
    
    
    loaded_data = loaded_data.reset_index(drop=True)
    
    return loaded_data, f"Current CSV: {current_file_name} - Shape: {loaded_data.shape}"

@tool
def add_column(input: AddColumnInput) -> str:
    """Add a new column to the loaded data with a default value."""
    global loaded_data
    try:
        loaded_data[input.column_name] = input.default_value
        return f"Added column '{input.column_name}'. New columns: {loaded_data.columns.tolist()}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def set_cell_value(input: SetCellValueInput) -> str:
    """Set the value of a specific cell in the loaded data."""
    global loaded_data
    try:
        
        row_index = int(input.row_index) if isinstance(input.row_index, float) else input.row_index
        
        
        column_name = input.column_name
        if column_name not in loaded_data.columns:
            existing_columns = loaded_data.columns.tolist()
            existing_columns_lower = [col.lower() for col in existing_columns]
            if column_name.lower() in existing_columns_lower:
                column_name = existing_columns[existing_columns_lower.index(column_name.lower())]
            else:
                return f"Error: Column '{input.column_name}' not found. Available columns: {loaded_data.columns.tolist()}"
        
        loaded_data.at[row_index, column_name] = input.value
        return f"Set value at row {row_index}, column '{column_name}' to {input.value}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def set_row_values(input: SetRowValuesInput) -> str:
    """Set multiple values for a specific row in the loaded data."""
    global loaded_data
    try:
        if loaded_data is None:
            return "No data loaded."
            
        
        values = input.values
        if isinstance(values, str):
            import json
            try:
                values = json.loads(values)
            except json.JSONDecodeError as e:
                return f"Error parsing values: {str(e)}. Please provide a valid JSON object."
        
       
        if not isinstance(values, dict):
            return f"Error: values must be a dictionary or a JSON string representing a dictionary, got {type(values)}"
        
      
        row_index = input.row_index
        if isinstance(row_index, str):
      
            name_col = None
            for col in loaded_data.columns:
                if col.lower() in ['name', 'title', 'id']:
                    name_col = col
                    break
            
            if name_col:
                matches = loaded_data[loaded_data[name_col].astype(str).str.lower() == row_index.lower()]
                if len(matches) > 0:
                    row_index = matches.index[0]
                else:
                    return f"No row found with name '{row_index}'"
            else:
                return f"Cannot find row by name - no suitable name column found"
        elif isinstance(row_index, float):
            row_index = int(row_index)
        
        
        if row_index < 0 or row_index >= len(loaded_data):
            return f"Error: Row index {row_index} is out of bounds. Valid indices are 0 to {len(loaded_data)-1}."
        
       
        normalized_values = {}
        existing_columns = loaded_data.columns.tolist()
        existing_columns_lower = [col.lower() for col in existing_columns]
        
        for key, value in values.items():
          
            if key in existing_columns:
                normalized_values[key] = value
       
            elif key.lower() in existing_columns_lower:
                correct_key = existing_columns[existing_columns_lower.index(key.lower())]
                normalized_values[correct_key] = value
            else:
              
                normalized_values[key] = value
        
    
        for col, value in normalized_values.items():
            loaded_data.at[row_index, col] = value
            
        return f"Set values for row {row_index}: {list(normalized_values.keys())}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def list_columns() -> str:
    """List all column names in the loaded CSV data."""
    global loaded_data
    if loaded_data is None:
        return "No data loaded."
    return f"Available columns: {loaded_data.columns.tolist()}"

tools = [
    load_csv,
    show_data,
    remove_rows,
    remove_columns,
    add_row,
    add_column,
    set_cell_value,
    set_row_values,
    list_columns,
    get_data_info,
    find_row_index  
]


model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a CSV data manipulation assistant. 
    When working with column names, be flexible with capitalization and try to match column names in a case-insensitive way.
    Always check the available columns before performing operations and suggest corrections if the user's input doesn't exactly match the column names."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_tools_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Gradio interface
def chatbot_interface(message, history):
  
    langchain_history = []
    for human_msg, ai_msg in history:
        if human_msg and ai_msg:  # Only add non-empty messages
            langchain_history.append(HumanMessage(content=human_msg))
            langchain_history.append(AIMessage(content=ai_msg))
    
    
    if not message or message.strip() == "":
        return "Please provide a valid command or question."
    
    try:
        
        response = agent_executor.invoke({
            "input": message, 
            "chat_history": langchain_history
        })
        return response["output"]
    except Exception as e:
        # To handle API errors gracefully
        error_msg = str(e)
        if "empty text parameter" in error_msg:
            return "I couldn't process that request. Please try rephrasing your command."
        else:
            return f"An error occurred: {error_msg}"

def display_csv():
    global loaded_data
    if loaded_data is None:
        return pd.DataFrame(), "No CSV loaded yet."
    return loaded_data, f"Current CSV: {current_file_name} - Shape: {loaded_data.shape}"

def process_upload(file):
    global loaded_data, current_file_name
    try:
        file_path = file.name
        loaded_data = pd.read_csv(file_path)
        current_file_name = file_path
        return loaded_data, f"Loaded CSV: {current_file_name} - Shape: {loaded_data.shape}"
    except Exception as e:
        return pd.DataFrame(), f"Error loading file: {str(e)}"

with gr.Blocks(title="Gemini-Powered CSV Agent") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload CSV File")
            status = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            csv_display = gr.Dataframe(label="CSV Data")
            
    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation")
        
    with gr.Row():
        msg = gr.Textbox(label="Command", placeholder="Type your command")
        send = gr.Button("Send")
        
    with gr.Row():
        refresh_btn = gr.Button("Refresh CSV View")
        download_btn = gr.Button("Download Modified CSV")
    
    file_upload.upload(process_upload, inputs=[file_upload], outputs=[csv_display, status])
    
    refresh_btn.click(display_csv, inputs=[], outputs=[csv_display, status])
    
    def download_csv():
        global loaded_data, current_file_name
        if loaded_data is None:
            return None
        
        if current_file_name:
            base_name = os.path.basename(current_file_name)
            name, ext = os.path.splitext(base_name)
            download_name = f"{name}_modified{ext}"
        else:
            download_name = "modified_data.csv"
            
        temp_path = os.path.join(os.getcwd(), download_name)
        loaded_data.to_csv(temp_path, index=False)
        return temp_path
    
    download_btn.click(download_csv, inputs=[], outputs=[gr.File(label="Download")])
    
    def respond(message, chat_history):
        bot_message = chatbot_interface(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send.click(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()