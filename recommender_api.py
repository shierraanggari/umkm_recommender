from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import traceback # debugging

app = Flask(__name__)

stemmer = None
stopword_remover = None
SASTRAWI_INIT_SUCCESS = False
try:
    print(f"[FLASK_API_LOG] Mencoba inisialisasi Sastrawi...")
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    SASTRAWI_INIT_SUCCESS = True
    print("[FLASK_API_LOG] Sastrawi BERHASIL diinisialisasi.")
except Exception as e_sastrawi:
    print(f"[FLASK_API_LOG] PERINGATAN: Gagal inisialisasi Sastrawi: {e_sastrawi}")
    print(f"[FLASK_API_LOG] Traceback Sastrawi Init: {traceback.format_exc()}")

# Preprocessing dan Rekomendasi
def preprocess_text(text_input):
    # print(f"[FLASK_API_LOG] Preprocessing: {text_input[:30] if isinstance(text_input, str) else 'Input bukan string'}")
    if not isinstance(text_input, str):
        return ""
    text = text_input.lower()
    if SASTRAWI_INIT_SUCCESS and stopword_remover:
        text = stopword_remover.remove(text)
    if SASTRAWI_INIT_SUCCESS and stemmer:
        text = stemmer.stem(text)
    return text

def get_recommendations(current_lapak, other_lapaks_list, top_n=5):
    recommendations = []
    # print(f"[FLASK_API_LOG] get_recommendations dipanggil. Current ID: {current_lapak.get('id') if current_lapak else 'N/A'}, Others: {len(other_lapaks_list) if other_lapaks_list else 0}")
    try:
        if not current_lapak or not isinstance(current_lapak, dict) or \
           not other_lapaks_list or not isinstance(other_lapaks_list, list):
            print("[FLASK_API_LOG] Data input tidak valid untuk get_recommendations.")
            return []

        corpus_items = [current_lapak] + other_lapaks_list
        if len(corpus_items) < 2:
            print("[FLASK_API_LOG] Korpus kurang dari 2 item.")
            return []

        processed_texts = []
        for item in corpus_items:
            nama = item.get('name', '')
            deskripsi = item.get('description_raw', item.get('description', ''))
            tipe = item.get('type_label', '')
            kecamatan = item.get('kecamatan_label', '')
            kelurahan = item.get('kelurahan_label', '')

            teks_gabungan = f"{str(nama)} {str(nama)} {str(deskripsi)} {str(tipe)} {str(kecamatan)} {str(kelurahan)} {str(kelurahan)}"
            processed_texts.append(preprocess_text(teks_gabungan))
        
        if not any(processed_texts):
            print("[FLASK_API_LOG] Semua teks kosong setelah preprocessing.")
            return []

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        if tfidf_matrix.shape[0] <= 1:
            print("[FLASK_API_LOG] Matriks TF-IDF tidak cukup baris.")
            return []
            
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        if cosine_similarities.size == 0:
            print("[FLASK_API_LOG] Cosine similarities kosong.")
            return []

        similarity_scores = []
        for i, other_item in enumerate(other_lapaks_list):
            score = cosine_similarities[0, i]
            if score > 0.01:
                processed_text_for_item = processed_texts[i + 1]
                similarity_scores.append({
                    'id': str(other_item.get('id', f'UNKNOWN_ID_{i}')),
                    'score': score,
                    'preprocessed_text': processed_text_for_item
                })
        
        sorted_items = sorted(similarity_scores, key=lambda x: x['score'], reverse=True)
        recommendations = [{'id': item_data['id'], 'score': item_data['score']} for item_data in sorted_items[:top_n]]
        
        preprocessed_text_utama = processed_texts[0]
        print(f"  > PEMBANDING (ID: {current_lapak.get('id')}): {preprocessed_text_utama}")
        print("  --------------------------------------------------")

        print("[FLASK_API_LOG] --- Detail Rekomendasi (Untuk Analisis) ---")
        for item_data in sorted_items[:top_n]:
            print(f"  - ID: {item_data['id']}, Skor: {item_data['score']:.4f}, Teks: {item_data['preprocessed_text']}")
        print("[FLASK_API_LOG] -----------------------------------------")

        print(f"[FLASK_API_LOG] Rekomendasi dihasilkan: {recommendations}")

    except Exception as e_rec:
        print(f"[FLASK_API_LOG] ERROR dalam get_recommendations: {e_rec}")
        print(f"[FLASK_API_LOG] Traceback get_recommendations: {traceback.format_exc()}")
    return recommendations

# Endpoint API
@app.route('/recommend', methods=['POST'])
def recommend_route():
    print("[FLASK_API_LOG] Request diterima di endpoint /recommend")
    try:
        data = request.get_json()
        if not data:
            print("[FLASK_API_LOG] Tidak ada data JSON di request body.")
            return jsonify({"error": "Request body harus JSON"}), 400

        current_lapak = data.get('current_lapak')
        all_other_lapaks = data.get('all_other_lapaks')
        top_n = int(data.get('top_n', 5))

        if not current_lapak or not all_other_lapaks:
            print("[FLASK_API_LOG] 'current_lapak' atau 'all_other_lapaks' tidak ada di JSON input.")
            return jsonify({"error": "'current_lapak' dan 'all_other_lapaks' dibutuhkan"}), 400

        print(f"[FLASK_API_LOG] Data diterima untuk rekomendasi: Current ID {current_lapak.get('id')}, Others count: {len(all_other_lapaks)}")
        
        recommended_ids = get_recommendations(current_lapak, all_other_lapaks, top_n)
        
        print(f"[FLASK_API_LOG] Mengirim respons: {recommended_ids}")
        return jsonify({"recommendations": recommended_ids, "error": None})

    except Exception as e_api:
        print(f"[FLASK_API_LOG] ERROR di endpoint /recommend: {e_api}")
        print(f"[FLASK_API_LOG] Traceback API: {traceback.format_exc()}")
        return jsonify({"error": f"Error internal server API Python: {str(e_api)}", "recommendations": []}), 500

if __name__ == '__main__':
    print("[FLASK_API_LOG] Memulai Flask server...")
    app.run(host='127.0.0.1', port=5001, debug=False)