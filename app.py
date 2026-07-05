import streamlit as st
import pandas as pd
import time
import re
from google import genai

st.set_page_config(page_title="Mbsky Store - Rakit PC Anda!", page_icon="🖥️", layout="centered")

# ── RATE LIMIT ──────────────────────────────────────────────────────────────────

MAX_REQ_PER_MINUTE = 10
WINDOW = 60

def check_rate_limit():
    now = time.time()
    timestamps = st.session_state.get("request_timestamps", [])
    timestamps = [t for t in timestamps if now - t < WINDOW]
    st.session_state.request_timestamps = timestamps

    if len(timestamps) >= MAX_REQ_PER_MINUTE:
        wait = int(WINDOW - (now - timestamps[0]))
        return False, wait
    return True, 0

# ── CONTEXT GUARD ──────────────────────────────────────────────────────────────

PC_KEYWORDS = [
    "pc", "komputer", "cpu", "processor", "motherboard", "ram", "memory",
    "vga", "gpu", "graphic", "grafis", "storage", "ssd", "hdd", "psu",
    "power supply", "casing", "case", "rakit", "build", "komponen",
    "part", "upgrade", "gaming", "budget", "harga", "rekomendasi",
    "rekomendasi", "kompatibel", "compatible", "processor", "core i",
    "ryzen", "intel", "amd", "nvidia", "geforce", "radeon", "ddr4",
    "ddr5", "m.2", "pcie", "atx", "micro-atx", "mini-itx", "watt",
    "cooler", "pendingin", "thermal", "bottleneck", "benchmark", "fps",
    "dollar", "jutaan", "juta", "spek", "spesifikasi", "toko",
]

OFFTOPIC_PATTERNS = [
    r'\b(resep|masak|makanan|minuman|kue)\b',
    r'\b(cinta|pacaran|putus|jodoh|nikah)\b',
    r'\b(politik|presiden|pemilu|partai|gubernur)\b',
    r'\b(agama|tuhan|surga|neraka|dosa|ibadah)\b',
    r'\b(obat|penyakit|rumah sakit|dokter|sakit)\b',
    r'\b(lirik|lagu|musik|band|penyanyi|album)\b',
    r'\b(film|drama|sinetron|aktor|actress)\b',
    r'\b(soal|jawaban|pr|tugas|ujian|sekolah)\b',
    r'\b(cerita|novel|puisi|pantun|dongeng)\b',
    r'\b(olahraga|sepak bola|basket|bulu tangkis)\b',
    r'\b(hewan|kucing|anjing|burung|ikan)\b',
    r'\b(tanam|kebun|pertanian|pupuk)\b',
    r'\b(cuaca|iklim|hujan|panas|gempa)\b',
]

def is_allowed(prompt):
    prompt_lower = prompt.lower().strip()

    greets = re.match(
        r'^(halo|hai|hi|hey|helo|pagi|siang|sore|malam|makasih|terima kasih'
        r'|thanks|ok|oke|iya|ya|tidak|ga|nggak|gak|test|tes)$',
        prompt_lower
    )
    if greets:
        return True

    has_pc_keyword = any(kw in prompt_lower for kw in PC_KEYWORDS)
    has_offtopic = any(re.search(p, prompt_lower) for p in OFFTOPIC_PATTERNS)

    if has_pc_keyword:
        return True
    if has_offtopic:
        return False

    if len(prompt_lower.split()) <= 6:
        for msg in reversed(st.session_state.get("messages", [])[-4:]):
            if msg["role"] == "assistant" and ("PC" in msg["content"]
               or "rekomendasi" in msg["content"].lower()
               or "komponen" in msg["content"].lower()
               or "total" in msg["content"].lower()):
                return True
        return False

    return None  # biar Gemini yang mutusin

# ── DATA ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv("data_komponen_pc.csv")

@st.cache_data
def format_data(df):
    lines = []
    for cat in df["Kategori"].unique():
        lines.append(f"\n─── {cat.upper()} ───")
        subset = df[df["Kategori"] == cat]
        for _, r in subset.iterrows():
            harga = f"Rp{r['Harga']:,}"
            lines.append(
                f"- {r['ID']}: {r['Nama_Produk']} | {r['Spesifikasi']} | "
                f"{harga} | Kompatibilitas: {r['Kompatibilitas']}"
            )
    return "\n".join(lines)

df = load_data()
DATA_STR = format_data(df)

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""Kamu adalah asisten spesialis perakitan PC untuk toko "Mbsky Store - Rakit PC Anda!".
Tugasmu SATU-SATUNYA adalah membantu pengguna memilih dan merakit komponen PC.

── BATASAN KETAT ──
- Kamu HANYA boleh menjawab pertanyaan seputar perakitan PC, komponen komputer, dan rekomendasi produk dari toko.
- Jika user bertanya di luar topik (politik, agama, resep masakan, kesehatan, hiburan, dll), KAMU WAJIB MENOLAK dengan sopan.
- Contoh penolakan: "Maaf, saya hanya bisa membantu seputar perakitan PC dan produk toko kami. Ada yang bisa saya bantu terkait PC?"
- HANYA rekomendasikan produk yang ADA di data toko di bawah ini.
- JANGAN pernah merekomendasikan produk di luar data.
- Jika budget tidak cukup untuk build yang layak, beri tahu dengan jujur.

── PANDUAN KOMPATIBILITAS ──
1. CPU & Motherboard → Socket harus cocok (LGA1700, AM4, AM5)
2. RAM & Motherboard → Tipe DDR harus cocok (DDR4/DDR5)
3. VGA → Semua motherboard support PCIe (universal)
4. Storage M.2 → Butuh slot M.2 di motherboard
5. PSU → Standar ATX, pastikan watt cukup untuk semua komponen
6. Casing → Pastikan ukuran motherboard didukung (Micro-ATX / ATX)

── FORMAT RESPON ──
- Berikan rekomendasi LENGKAP: CPU, Motherboard, RAM, VGA, Storage, PSU, Casing
- Tampilkan harga SATUAN dan TOTAL
- Jika budget terbatas, beri opsi alternatif yang lebih murah
- Gunakan bahasa Indonesia yang santun, ramah, dan informatif

── DATA PRODUK TOKO ──
{DATA_STR}
"""

# ── API KEY ────────────────────────────────────────────────────────────────────

def get_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return st.session_state.get("api_key", "")

API_KEY = get_api_key()

if not API_KEY:
    st.markdown(
        "<h1 style='text-align:center; color:#00d2ff;'>🖥️ Mbsky Store - Rakit PC Anda!</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Masukkan <strong>Google AI Studio API Key</strong> "
        "untuk memulai.</p>",
        unsafe_allow_html=True,
    )
    with st.expander("📋 Cara dapat API Key (gratis)", expanded=True):
        st.markdown("""
        1. Buka [Google AI Studio](https://aistudio.google.com/apikey)
        2. Login dengan Google Account
        3. Klik **"Create API Key"**
        4. Salin key-nya (diawali `AIza...`)
        """)
    key = st.text_input("API Key", type="password", placeholder="AIza...")
    if st.button("🚀 Mulai", type="primary", use_container_width=True):
        if key.startswith("AIza"):
            st.session_state.api_key = key
            st.rerun()
        else:
            st.error("Key tidak valid. Pastikan diawali 'AIza...'")
    st.stop()

# ── GEMINI ─────────────────────────────────────────────────────────────────────

client = genai.Client(api_key=API_KEY)

# ── STATE ──────────────────────────────────────────────────────────────────────

if "request_timestamps" not in st.session_state:
    st.session_state.request_timestamps = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Halo! 🖥️ Saya asisten **Mbsky Store - Rakit PC Anda!**.\n\n"
                "Ceritakan kebutuhan PC-mu:\n"
                "• 💰 **Budget** berapa?\n"
                "• 🎯 Mau dipakai untuk **(gaming, office, editing, desain)**?\n"
                "• 🏷️ Ada preferensi merk tertentu?\n\n"
                "Atau klik salah satu pertanyaan di bawah 👇"
            ),
        }
    ]

# ── UI HEADER ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; }
    .header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .header h1 {
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        margin: 0;
    }
    .header p {
        color: #888;
        margin: 0;
        font-size: 0.9rem;
    }
    .suggestion-btn button {
        background: #1a1a2e !important;
        border: 1px solid #333 !important;
        color: #ccc !important;
        font-size: 0.85rem !important;
    }
    .suggestion-btn button:hover {
        border-color: #00d2ff !important;
        color: #00d2ff !important;
    }
    footer { visibility: hidden; }
</style>
<div class="header">
    <h1>🖥️ Mbsky Store - Rakit PC Anda!</h1>
    <p>💡 Asisten perakitan PC dari toko kami</p>
</div>
""", unsafe_allow_html=True)

# ── RATE LIMIT INDICATOR ────────────────────────────────────────────────────────

rate_ok, wait_time = check_rate_limit()
if not rate_ok:
    st.warning(f"⏳ Terlalu banyak permintaan. Tunggu {wait_time} detik.")

remaining = MAX_REQ_PER_MINUTE - len(st.session_state.get("request_timestamps", []))
st.caption(f"💬 Sisa percakapan: {remaining}/{MAX_REQ_PER_MINUTE} per menit")

# ── CHAT ───────────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def generate_response(prompt):
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text

# ── SUGGESTED QUESTIONS ────────────────────────────────────────────────────────

if len(st.session_state.messages) <= 1:
    st.markdown("##### 🔍 Coba tanya:")
    cols = st.columns(2)
    suggestions = [
        "🎮 PC gaming budget 8 juta",
        "💼 PC kantoran 5 jutaan",
        "🎬 PC editing video 10jt",
        "🔄 Upgrade PC lama saya",
    ]
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, use_container_width=True, key=f"sug_{i}"):
            st.session_state.pending = s
            st.rerun()

def process_prompt(prompt):
    verdict = is_allowed(prompt)
    if verdict is False:
        return (
            "Maaf, saya hanya bisa membantu seputar perakitan PC dan "
            "produk toko kami. 🤖\n\n"
            "Silakan tanyakan tentang rakit PC, rekomendasi komponen, "
            "atau harga produk!"
        )

    rate_ok, wait_time = check_rate_limit()
    if not rate_ok:
        st.rerun()

    st.session_state.request_timestamps.append(time.time())
    return st.write_stream(generate_response(prompt))

# Handle pending suggestion from buttons
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = process_prompt(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("Tanyakan tentang rakit PC..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = process_prompt(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── FOOTER ─────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center;color:#555;font-size:0.75rem;'>"
    "© Mbsky Store. Data produk berdasarkan stok toko. Harga dapat berubah sewaktu-waktu.<br>"
    "Powered by Google Gemini &nbsp;·&nbsp; "
    "<a href='https://streamlit.io' style='color:#555;'>Streamlit</a></p>",
    unsafe_allow_html=True,
)
