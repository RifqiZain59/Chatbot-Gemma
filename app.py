from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from pyngrok import ngrok

app = Flask(__name__)

# --- 0. LOGIN HUGGING FACE ---
# JANGAN LUPA: Ganti dengan token barumu!
login(token=HF_TOKEN)

# --- 1. TENTUKAN PATH SESUAI FOLDER ---
BASE_MODEL_ID = "google/gemma-3-4b-it" 
ADAPTER_DIR = "./gemma-3-wayang-final" 

print("Memuat model ke memori... (Ini butuh waktu beberapa saat)")

# --- 2. SETUP MEMORI ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, 
)

# --- 3. LOAD BASE MODEL & TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# --- 4. GABUNGKAN BASE MODEL DENGAN ADAPTER ---
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# [KUNCI KECEPATAN 1] Kunci model ke mode Evaluasi (bukan mode Training)
model.eval() 
print("Model berhasil dimuat dan siap digunakan!")

# --- 5. ENDPOINT API ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "Pesan tidak boleh kosong"}), 400

        messages = [
            {"role": "user", "content": user_message}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        # [KUNCI KECEPATAN 2] Matikan fitur kalkulasi training PyTorch
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200, # <--- INI YANG DIUBAH (sebelumnya 512, sekarang 200 agar lebih cepat dan tidak timeout)
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[1]
        response_token_ids = outputs[0][input_length:]
        response = tokenizer.decode(response_token_ids, skip_special_tokens=True)

        return jsonify({"response": response})

    except Exception as e:
        # Jika terjadi error di Python, error-nya akan dikirim ke Flutter
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = 5000
    
    # --- 6. SETUP NGROK DENGAN STATIC DOMAIN ---
    
    # ⚠️ PENTING: Untuk pakai domain statis, kamu WAJIB memasukkan Authtoken.
    # Dapatkan di dashboard ngrok kamu: https://dashboard.ngrok.com/get-started/your-authtoken
    NGROK_AUTH_TOKEN = "3A28h7PFtDgfl2Ins8NGkCkcQEH_269dSSArm2tfij2wjAtxZ"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    # Buka tunnel di port 5000 dengan domain spesifik yang kamu berikan
    try:
        public_url = ngrok.connect(
            port,
            domain="bifunctional-unstoutly-corrina.ngrok-free.dev"
        ).public_url
        
        print("=" * 60)
        print(f"🚀 Ngrok Tunnel Berhasil Dibuat!")
        print(f"🔗 URL API Publik kamu: {public_url}/chat")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Gagal menyambungkan ke Ngrok: {e}")
        print("Pastikan Authtoken benar dan domain tersebut sudah diklaim di akun Ngrok kamu.")
    
    # Jalankan aplikasi Flask
    app.run(host='0.0.0.0', port=port)