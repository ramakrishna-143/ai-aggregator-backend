import os
import requests
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Cross-Origin Resource Sharingని అనుమతిస్తుంది

# --- Hugging Face API Configuration ---
# Renderలో ఎన్విరాన్‌మెంట్ వేరియబుల్‌గా సెట్ చేసిన మీ Hugging Face API టోకెన్
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# మీరు Hugging Faceలో ఎంచుకున్న టెక్స్ట్ జనరేషన్ మోడల్ యొక్క Inference API URL
# ఉదాహరణకు: "https://api-inference.huggingface.co/models/google/gemma-7b-it"
TEXT_GEN_MODEL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"

# మీరు Hugging Faceలో ఎంచుకున్న ఇమేజ్ జనరేషన్ మోడల్ యొక్క Inference API URL
# ఉదాహరణకు: "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_GEN_MODEL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json" # Hugging Faceకి సాధారణంగా JSON అభ్యర్థనలు వెళ్తాయి
}

# --- Hugging Face APIకి అభ్యర్థన పంపే సహాయక ఫంక్షన్ ---
def query_hf_model(payload, model_url, is_image=False):
    if not HF_API_TOKEN:
        raise ValueError("Hugging Face API token is not configured.")

    try:
        if is_image:
            # ఇమేజ్ జనరేషన్ కోసం ప్రత్యేక హెడర్స్ లేదా డేటా ఫార్మాట్ అవసరం కావచ్చు
            # కొన్ని మోడల్స్ JSONని ఆశిస్తే, కొన్ని డైరెక్ట్ స్ట్రింగ్ లేదా ఫైల్ వంటివి ఆశించవచ్చు
            # ఇక్కడ అత్యంత సాధారణ JSON payload ఉదాహరణ ఇస్తున్నాం
            response = requests.post(model_url, headers=headers, json=payload, timeout=120) # ఇమేజ్ జనరేషన్ సమయం పట్టవచ్చు
            if response.status_code == 200:
                return response.content # ఇమేజ్ బైట్‌లు
            else:
                print(f"Image generation failed with status {response.status_code}: {response.text}")
                return None # లేదా ఎర్రర్‌ను తిరిగి ఇవ్వండి
        else:
            response = requests.post(model_url, headers=headers, json=payload, timeout=60) # టెక్స్ట్ జనరేషన్
            response.raise_for_status() # HTTP ఎర్రర్‌ల కోసం Ausnahmeని పెంచండి
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request to Hugging Face model failed: {e}")
        raise

# --- ఫ్రంటెండ్ నుండి వచ్చే అభ్యర్థనలను ప్రాసెస్ చేసే రూట్ ---
@app.route('/ai-tool-proxy', methods=['POST'])
def ai_tool_proxy():
    if not HF_API_TOKEN:
        return jsonify({"error": "Server error: Hugging Face API token not configured."}), 500

    data = request.json
    tool = data.get('tool')
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        if tool == "text_generation":
            payload = {"inputs": prompt}
            output = query_hf_model(payload, TEXT_GEN_MODEL)

            if isinstance(output, list) and output and 'generated_text' in output[0]:
                generated_text = output[0]['generated_text']
                return jsonify({"generatedText": generated_text}), 200
            else:
                return jsonify({"error": "Failed to generate text from model.", "details": output}), 500

        elif tool == "image_generation":
            payload = {"inputs": prompt}
            image_bytes = query_hf_model(payload, IMAGE_GEN_MODEL, is_image=True)

            if image_bytes:
                # ఇమేజ్ బైట్‌లను base64లోకి ఎన్కోడ్ చేయండి
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                # బ్రౌజర్‌లో డిస్ప్లే చేయడానికి data URIని తిరిగి ఇవ్వండి
                return jsonify({"imageUrl": f"data:image/jpeg;base64,{base64_image}"}), 200
            else:
                return jsonify({"error": "Failed to generate image from model."}), 500

        else:
            return jsonify({"error": "Invalid AI tool specified."}), 400

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

# --- అప్లికేషన్‌ను రన్ చేయండి (డెవలప్‌మెంట్ కోసం) ---
if __name__ == '__main__':
    # Renderలో gunicorn ద్వారా అప్లికేషన్ రన్ అవుతుంది, కాబట్టి ఈ భాగం డెవలప్‌మెంట్ కోసం మాత్రమే.
    # Productionలో debug=True ఉండకూడదు.
    app.run(debug=True, port=os.environ.get("PORT", 5000))
