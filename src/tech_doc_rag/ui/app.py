import requests
import streamlit as st

# API Endpoint Konfigürasyonu
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Technical Doc RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- SİDEBAR: DOKÜMAN YÜKLEME (INGESTION) ---
with st.sidebar:
    st.header("📚 Doküman Yönetimi")
    st.markdown("Vektör veritabanına yeni teknik dokümanlar ekleyin.")

    tab1, tab2 = st.tabs(["Dosya Yükle", "Metin Yapıştır"])

    with tab1:
        uploaded_file = st.file_uploader(
            ".md veya .txt dosyası seçin", 
            type=["md", "txt"],
            key="file_uploader_widget"
        )
        if uploaded_file and st.button("Dosyayı İndeksle", use_container_width=True, key="btn_ingest_file"):
            with st.spinner("Doküman parçalanıyor ve vektörleştiriliyor..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/markdown")}
                    res = requests.post(f"{API_BASE_URL}/ingest/file", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"✅ {data['message']} ({data['chunks_created']} parça eklendi)")
                    else:
                        st.error(f"Hata: {res.json().get('detail', 'Yükleme başarısız')}")
                except Exception as e:
                    st.error(f"API Bağlantı Hatası: {e}")

    with tab2:
        raw_text = st.text_area(
            "Markdown Metni", 
            height=150, 
            placeholder="# Başlık\nMetin içeriği...",
            key="txt_area_raw_text"
        )
        doc_name = st.text_input(
            "Doküman Adı", 
            value="custom_doc.md",
            key="txt_input_doc_name"
        )
        if st.button("Metni İndeksle", use_container_width=True, key="btn_ingest_text"):
            if raw_text.strip():
                with st.spinner("Metin işleniyor..."):
                    try:
                        payload = {"content": raw_text, "source_file": doc_name}
                        res = requests.post(f"{API_BASE_URL}/ingest/text", json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            st.success(f"✅ {data['message']} ({data['chunks_created']} parça eklendi)")
                        else:
                            st.error(f"Hata: {res.json().get('detail', 'İşlem başarısız')}")
                    except Exception as e:
                        st.error(f"API Bağlantı Hatası: {e}")
            else:
                st.warning("Lütfen metin giriniz.")

    st.divider()
    st.markdown("### ⚙️ Ayarlar")
    top_k = st.slider(
        "Getirilecek Parça Sayısı (top_k)", 
        min_value=1, 
        max_value=5, 
        value=2,
        key="slider_top_k"
    )
    similarity_threshold = st.slider(
        "Benzerlik Eşiği", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.2, 
        step=0.05,
        key="slider_similarity_threshold"
    )


# --- ANA EKRAN: SOHBET ARAYÜZÜ ---
st.title("🤖 Technical Documentation RAG Assistant")
st.caption("Markdown dokümanlarınız üzerinden kaynak gösterimli (Citation) akıllı yanıtlar alın.")

# Chat Geçmişini Session State'te Tutma
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş Mesajları Ekrana Basma
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("📚 Kullanılan Kaynaklar"):
                for idx, cite in enumerate(msg["citations"], 1):
                    st.markdown(f"**[{idx}] {cite['source_file']}** -> `{cite['section_header']}`")
                    st.caption(cite["snippet"])

# Kullanıcı Soru Girişi
if user_query := st.chat_input("FastAPI hakkında bir soru sorun...", key="chat_input_query"):
    # Kullanıcı mesajını geçmişe ekle ve göster
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Asistan Yanıtını Hazırlama
    with st.chat_message("assistant"):
        with st.spinner("Dokümanlar taranıyor ve yanıt üretiliyor..."):
            try:
                payload = {
                    "query": user_query,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold
                }
                response = requests.post(f"{API_BASE_URL}/query", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    citations = data.get("citations", [])

                    st.markdown(answer)

                    if citations:
                        with st.expander("📚 Kullanılan Kaynaklar"):
                            for idx, cite in enumerate(citations, 1):
                                st.markdown(f"**[{idx}] {cite['source_file']}** -> `{cite['section_header']}`")
                                st.caption(cite["snippet"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })
                else:
                    st.error("API'den yanıt alınamadı.")
            except Exception as e:
                st.error(f"API ile iletişim kurulamadı: {e}")