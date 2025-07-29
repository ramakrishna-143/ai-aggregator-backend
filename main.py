# Import necessary libraries
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all origins, allowing your frontend to communicate with this backend
CORS(app)

# IMPORTANT: Replace with your actual Google Cloud API Key
# It is highly recommended to store this securely, e.g., in environment variables
# For Google Cloud Functions, you can set environment variables directly.
# For local development, you can set it in your shell: export GOOGLE_API_KEY="YOUR_API_KEY"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Define the API endpoints for Google's AI models
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
IMAGEN_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict"

@app.route('/ai-tool-proxy', methods=['POST'])
def ai_tool_proxy():
    """
    Acts as a proxy for AI tool API calls.
    Receives requests from the frontend, determines the AI model to use
    based on 'tool_category', makes the appropriate API call to Google's
    AI services, and returns the result.
    """
    try:
        # Parse the incoming JSON request from the frontend
        data = request.get_json()
        tool_category = data.get('tool_category')
        prompt = data.get('prompt')

        if not tool_category or not prompt:
            # Return an error if essential data is missing
            return jsonify({"error": "Missing tool_category or prompt"}), 400

        # Initialize variables for API call
        api_url = ""
        payload = {}
        headers = {'Content-Type': 'application/json'}
        model_response_key = 'text' # Default for Gemini responses

        # Determine which AI model to call based on the tool_category
        if tool_category == 'text-generation':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        elif tool_category == 'image-generation':
            api_url = f"{IMAGEN_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"instances": {"prompt": prompt}, "parameters": {"sampleCount": 1}}
            model_response_key = 'image' # Special handling for Imagen image URL
        elif tool_category == 'refine-image-prompt':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Refine the following image generation prompt to make it more detailed and evocative, suitable for an AI image generator: \"{prompt}\""}]}]}
        elif tool_category == 'code-generation':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        elif tool_category == 'explain-code':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Explain the following code snippet in detail, including its purpose, how it works, and any potential improvements:\n```\n{prompt}\n```"}]}]}
        elif tool_category == 'speech-to-text':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Transcribe the following spoken input: \"{prompt}\""}]}]}
        elif tool_category == 'text-to-speech':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Generate speech for the following text: \"{prompt}\""}]}]}
        elif tool_category == 'suggest-tone':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Suggest a suitable tone (e.g., formal, casual, excited, calm) for the following text for text-to-speech conversion: \"{prompt}\""}]}]}
        elif tool_category == 'data-analysis':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Perform data analysis on: \"{prompt}\""}]}]}
        elif tool_category == 'generate-report-summary':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Generate a concise report summary from the following data and analysis: \"{prompt}\""}]}]}
        elif tool_category == 'translation':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Translate the following text: \"{prompt}\""}]}]}
        elif tool_category == 'detect-language':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Detect the language of the following text: \"{prompt}\""}]}]}
        elif tool_category == 'summarization':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Summarize the following text: \"{prompt}\""}]}]}
        elif tool_category == 'extract-keywords':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Extract key keywords from the following text: \"{prompt}\""}]}]}
        elif tool_category == 'chatbot':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        elif tool_category == 'summarize-conversation':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Summarize the following conversation snippet: \"{prompt}\""}]}]}
        elif tool_category == 'object-detection':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Perform object detection on: \"{prompt}\""}]}]}
        elif tool_category == 'image-captioning':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Generate a descriptive caption for an image that contains: \"{prompt}\""}]}]}
        elif tool_category == 'idea-brainstormer':
            api_url = f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"role": "user", "parts": [{"text": f"Brainstorm a list of creative ideas for: \"{prompt}\""}]}]}
        else:
            # Handle unsupported tool categories
            return jsonify({"error": "Unsupported tool category"}), 400

        # Make the API call to Google's AI service
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        api_result = response.json()

        # Extract the relevant part of the response based on the model
        if model_response_key == 'image':
            if api_result.get('predictions') and len(api_result['predictions']) > 0 and api_result['predictions'][0].get('bytesBase64Encoded'):
                # For image generation, return the base64 encoded image data
                return jsonify({"result": f"data:image/png;base64,{api_result['predictions'][0]['bytesBase64Encoded']}"})
            else:
                return jsonify({"error": "Image generation failed or returned no data"}), 500
        else:
            if api_result.get('candidates') and len(api_result['candidates']) > 0 and \
               api_result['candidates'][0].get('content') and api_result['candidates'][0]['content'].get('parts') and \
               len(api_result['candidates'][0]['content']['parts']) > 0:
                # For text-based models, return the generated text
                return jsonify({"result": api_result['candidates'][0]['content']['parts'][0]['text']})
            else:
                return jsonify({"error": "API response was unexpected or empty"}), 500

    except requests.exceptions.RequestException as e:
        # Handle network or API request errors
        app.logger.error(f"API request failed: {e}")
        return jsonify({"error": f"API request failed: {e}"}), 500
    except json.JSONDecodeError:
        # Handle invalid JSON in request
        return jsonify({"error": "Invalid JSON in request"}), 400
    except Exception as e:
        # Catch any other unexpected errors
        app.logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# Run the Flask app
if __name__ == '__main__':
    # For local development, run on port 8080
    # For Google Cloud Functions, the entry point is typically the function itself,
    # and Flask's run() is not directly used in the deployed environment.
    app.run(host='0.0.0.0', port=8080, debug=True)
