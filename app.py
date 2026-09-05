import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random

# --- Fix random seed for consistent results ---
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# --- AES S-Box ---
S_BOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
])

# Fixed round keys for rounds 2-10 (only round 1 is attacked)
FIXED_ROUND_KEYS = np.array([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab])

def hamming_weight(n):
    return bin(n).count('1')

def aes_full_power_trace(plaintext, first_round_key, with_countermeasure=False):
    """
    Simulates 10-round AES and returns power trace.
    The first round uses first_round_key, rounds 2-10 use fixed keys.
    """
    state = plaintext
    trace = []
    for r in range(10):
        if r == 0:
            key = first_round_key
        else:
            key = FIXED_ROUND_KEYS[r-1]
        
        xor_out = state ^ key
        sbox_out = S_BOX[xor_out]
        power = hamming_weight(sbox_out)
        
        if with_countermeasure:
            mask = np.random.randint(0, 256)
            masked = xor_out ^ mask
            power = hamming_weight(masked) + np.random.normal(0, 0.5)
        
        # Low noise for clear correlation
        power += np.random.normal(0, 0.02)
        trace.append(power)
        state = sbox_out
    return np.array(trace)

def run_cpa_attack_aes(cipher_type, secret_key, num_samples):
    """
    CPA attack on AES first round S-Box output.
    Targets: S_BOX(plaintext XOR first_round_key)
    """
    plaintexts = np.random.randint(0, 256, num_samples)
    protected = (cipher_type == "Protected")
    
    first_round_power = []
    for pt in plaintexts:
        trace = aes_full_power_trace(pt, secret_key, protected)
        first_round_power.append(trace[0])
    
    correlations = []
    for kg in range(256):
        intermediates = [S_BOX[pt ^ kg] for pt in plaintexts]
        predicted_power = [hamming_weight(iv) for iv in intermediates]
        corr = np.corrcoef(predicted_power, first_round_power)[0, 1]
        if np.isnan(corr):
            corr = 0
        correlations.append(corr)
    
    best_key = np.argmax(correlations)
    best_corr = max(correlations)
    return correlations, best_key, best_corr

def generate_ml_dataset(num_samples, protected=False):
    """
    Generates dataset for ML: full 10-round trace + target (plaintext XOR 0x42)
    """
    X, y = [], []
    for _ in range(num_samples):
        pt = np.random.randint(0, 256)
        trace = aes_full_power_trace(pt, first_round_key=0x42, with_countermeasure=protected)
        X.append(trace)
        y.append(pt ^ 0x42)
    return np.array(X), np.array(y)

def build_cnn_model(input_shape=(10,1)):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation='relu'),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_cnn(X, y, epochs=20, batch_size=64):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(-1, 10, 1)
    X_test = X_test.reshape(-1, 10, 1)
    y_train_cat = to_categorical(y_train, num_classes=256)
    y_test_cat = to_categorical(y_test, num_classes=256)
    model = build_cnn_model((10,1))
    model.fit(X_train, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    return acc

# --- Professional White Theme with Blue Accents ---
st.set_page_config(page_title="AES-10 Side-Channel Attack", layout="wide")

st.markdown("""
<style>
    /* Force white background */
    .stApp, .main, .block-container, .css-1d391kg {
        background-color: #ffffff !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    /* Buttons */
    .stButton > button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        color: white;
    }
    /* Metrics */
    .stMetric {
        background-color: #f8faff;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid #e5edf5;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    /* Headers */
    h1, h2, h3, h4, .st-emotion-cache-10trblm {
        color: #1e3a8a !important;
    }
    /* All text in markdown */
    .stMarkdown p, .stMarkdown li, .stMarkdown div {
        color: #1e3a8a !important;
    }
    /* Radio buttons */
    .stRadio label, .stRadio div, .stRadio span {
        color: #1e3a8a !important;
        font-weight: 500 !important;
    }
    /* Number input label visibility */
    .stNumberInput label {
        color: #1e3a8a !important;
        font-weight: 500 !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        color: #1e3a8a;
        background-color: #f0f4fa;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a !important;
        color: white !important;
    }
    /* Info messages */
    .stAlert {
        background-color: #f0f4fa !important;
        border-left-color: #1e3a8a !important;
    }
    .stAlert p {
        color: #1e3a8a !important;
    }
    /* Column text */
    .css-1v0mbdj p {
        color: #1e3a8a !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown("<h1 style='color: #1e3a8a;'>AES-10 Round Side-Channel Attack Simulator</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #2563eb; font-size: 1.2rem; font-weight: 400;'>Correlation Power Analysis (CPA) on AES First Round S-Box Output</p>", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2 = st.tabs(["CPA Attack", "Machine Learning"])

with tab1:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("<h3 style='color: #1e3a8a;'>Parameters</h3>", unsafe_allow_html=True)
        secret_key = st.number_input("Secret Key (0-255)", min_value=0, max_value=255, value=0x42, step=1)
        num_traces = st.slider("Number of Power Traces", min_value=100, max_value=10000, value=5000, step=100)
        cipher_type = st.radio("Cipher Type", ["Unprotected", "Protected"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run CPA Attack", use_container_width=True):
            with st.spinner("Analyzing AES traces..."):
                correlations, best_key, best_corr = run_cpa_attack_aes(cipher_type, secret_key, num_traces)
                if best_key == secret_key:
                    st.success(f"Key Recovered: {best_key} (Match)")
                else:
                    st.error(f"Key Recovered: {best_key} (Failed)")
                st.metric("Correlation Strength", f"{best_corr:.4f}")
                st.metric("Traces Used", num_traces)

    with col2:
        st.markdown("<h3 style='color: #1e3a8a;'>Correlation Plot</h3>", unsafe_allow_html=True)
        if 'correlations' in locals():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(256), correlations, 'b-', linewidth=1.5)
            ax.axvline(x=secret_key, color='r', linestyle='--', label=f'Real Key = {secret_key}')
            ax.set_xlabel("Key Guess", color='#1e3a8a')
            ax.set_ylabel("Correlation", color='#1e3a8a')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.spines['bottom'].set_color('#1e3a8a')
            ax.spines['left'].set_color('#1e3a8a')
            ax.tick_params(colors='#1e3a8a')
            st.pyplot(fig)
        else:
            st.info("Click 'Run CPA Attack' to see the correlation plot.")

with tab2:
    st.markdown("<h3 style='color: #1e3a8a;'>CNN Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #1e3a8a;'>Trains a Convolutional Neural Network on the full 10-round power trace to predict the first-round XOR intermediate value.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='color: #1e3a8a; font-weight: 500;'>Unprotected Data</p>", unsafe_allow_html=True)
        if st.button("Train CNN on Unprotected Traces", use_container_width=True):
            with st.spinner("Generating data and training CNN..."):
                X, y = generate_ml_dataset(2000, protected=False)
                acc = train_cnn(X, y, epochs=15, batch_size=64)
                st.success(f"Accuracy: {acc*100:.2f}%")
        else:
            st.info("Click the button to train CNN on unprotected AES-10 traces.")

    with col2:
        st.markdown("<p style='color: #1e3a8a; font-weight: 500;'>Protected Data</p>", unsafe_allow_html=True)
        if st.button("Train CNN on Protected Traces", use_container_width=True):
            with st.spinner("Generating data and training CNN..."):
                X, y = generate_ml_dataset(2000, protected=True)
                acc = train_cnn(X, y, epochs=15, batch_size=64)
                st.success(f"Accuracy: {acc*100:.2f}%")
        else:
            st.info("Click the button to train CNN on protected AES-10 traces.")