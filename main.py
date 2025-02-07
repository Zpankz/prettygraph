from typing import List
from pydantic import BaseModel
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import os
import json
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found in environment variables")

# Load the knowledge graph schema
with open('knowledge_graph_schema.json', 'r') as f:
    KNOWLEDGE_GRAPH_SCHEMA = json.load(f)['schema']

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # Preserve JSON order

class Node(BaseModel):
    id: str
    label: str

class Edge(BaseModel):
    source: str
    target: str
    label: str

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

# Color palette for nodes and edges
COLORS = [
    "#FF00FF", "#00FF00", "#FF0000", "#00FFFF", "#FF1493",
    "#7FFF00", "#FF69B4", "#39FF14", "#FF4D00", "#00BFFF",
    "#FF3131", "#40E0D0", "#FF1E8E", "#32CD32", "#FF0080",
    "#00FF7F", "#FF2400", "#00F5FF", "#FF00BF", "#7CFF01",
    "#FF3399", "#00FFB3", "#FF2D00", "#00FFCC", "#FF1493",
    "#39FF14", "#FF0066", "#00FFE5", "#FF4000", "#00FFFF"
]

def generate_knowledge_graph(text: str) -> KnowledgeGraph:
    """
    Generates a knowledge graph from the input text using OpenAI's structured outputs.
    """
    try:
        logger.debug(f"Generating knowledge graph for text: {text}")
        
        completion = client.beta.chat.completions.parse(
            model="o3-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI expert specializing in knowledge graph creation.
                    Create a knowledge graph based on the input text, where:
                    - Node labels must be direct words/phrases from the input
                    - Edge labels must be direct words/phrases from the input
                    - All source and target IDs in edges must match existing node IDs
                    - Node IDs should be simple alphanumeric strings
                    - Each node should have a unique ID"""
                },
                {"role": "user", "content": text}
            ],
            response_format=KnowledgeGraph
        )
        
        # Parse response
        message = completion.choices[0].message
        if message.parsed:
            graph_data = message.parsed
            logger.debug(f"Parsed graph data: {graph_data}")
            return graph_data
        else:
            logger.error("Model refused to generate a response")
            raise ValueError("Model refused to generate a response")
    
    except Exception as e:
        logger.error(f"Error in generate_knowledge_graph: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_graph', methods=['POST'])
def update_graph():
    try:
        data = request.get_json()
        text_content = data.get('text', '')
        
        if not text_content.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Generate knowledge graph
        graph_data = generate_knowledge_graph(text_content)
        
        # Convert to force-graph format
        nodes = []
        links = []  
        color_map = {}
        color_idx = 0
        
        # Process nodes
        for node in graph_data.nodes:
            if node.id not in color_map:
                color_map[node.id] = COLORS[color_idx % len(COLORS)]
                color_idx += 1
            
            nodes.append({
                'id': node.id,
                'label': node.label,
                'color': color_map[node.id]
            })
        
        # Process links (formerly edges)
        for edge in graph_data.edges:
            links.append({
                'source': edge.source,
                'target': edge.target,
                'label': edge.label,
                'color': COLORS[color_idx % len(COLORS)]
            })
            color_idx += 1
        
        # Final force-graph format
        force_graph_data = {
            'nodes': nodes,
            'links': links  
        }
        
        logger.debug(f"Force graph format: {force_graph_data}")
        return jsonify(force_graph_data)
        
    except Exception as e:
        logger.error(f"Error in update_graph: {str(e)}", exc_info=True)  
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', '1234'))
    app.run(host='0.0.0.0', port=port, debug=True)
