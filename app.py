import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import tempfile
import re
import numpy as np
import uuid 
import hashlib
from src.agents.ingestion import csv_ingestion
from src.agents.document_ingestion import document_ingestion
from src.agents.analytical_agent import analytical_agent

# Page config
st.set_page_config(
    page_title="Trust First AI Analyst",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern AI Chat Interface CSS
st.markdown("""
<style>
/* =============================================================================
   📁 FILE: assets/styles/main.css or styles/chat_interface.css
   🎨 MODERN AI CHAT INTERFACE - ENHANCED DESIGN SYSTEM
   ============================================================================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* =============================================================================
   🎯 CSS CUSTOM PROPERTIES (DESIGN TOKENS)
   ============================================================================= */

:root {
    /* Color Palette - Dark Theme */
    --color-bg-main: #0f1116;
    --color-bg-secondary: #1a1c23;
    --color-bg-tertiary: #252830;
    --color-bg-hover: #2d3038;
    --color-bg-elevated: #32353e;
    
    /* Border & Dividers */
    --color-border: #2f323b;
    --color-border-light: #383b45;
    --color-border-accent: rgba(99, 102, 241, 0.3);
    
    /* Accent Colors */
    --color-accent: #6366f1;
    --color-accent-hover: #4f46e5;
    --color-accent-light: rgba(99, 102, 241, 0.15);
    
    /* Text Colors */
    --color-text-primary: #f9fafb;
    --color-text-secondary: #9ca3af;
    --color-text-muted: #6b7280;
    --color-text-disabled: #4b5563;
    
    /* Semantic Colors - Success */
    --color-success-bg: #064e3b;
    --color-success-border: #22c55e;
    --color-success-text: #bbf7d0;
    
    /* Semantic Colors - Info */
    --color-info-bg: #1e3a8a;
    --color-info-border: #3b82f6;
    --color-info-text: #bfdbfe;
    
    /* Semantic Colors - Error */
    --color-error-bg: #450a0a;
    --color-error-border: #ef4444;
    --color-error-text: #fecaca;
    
    /* Semantic Colors - Warning */
    --color-warning-bg: #451a03;
    --color-warning-border: #f59e0b;
    --color-warning-text: #fde68a;
    
    /* Spacing Scale */
    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    --spacing-2xl: 3rem;
    
    /* Border Radius */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 18px;
    --radius-full: 9999px;
    
    /* Shadows */
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 2px 6px rgba(0, 0, 0, 0.35);
    --shadow-lg: 0 4px 12px rgba(0, 0, 0, 0.4);
    --shadow-accent: 0 4px 12px rgba(99, 102, 241, 0.3);
    
    /* Transitions */
    --transition-fast: 0.15s ease-in-out;
    --transition-base: 0.25s ease-in-out;
    --transition-slow: 0.35s ease-in-out;
    
    /* Typography */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --line-height-tight: 1.4;
    --line-height-normal: 1.6;
    --line-height-relaxed: 1.8;
}

/* =============================================================================
   🌐 GLOBAL RESET & BASE STYLES
   ============================================================================= */

* {
    font-family: var(--font-family);
    color: var(--color-text-primary);
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

body {
    margin: 0;
    padding: 0;
    background: var(--color-bg-main);
    overflow-x: hidden;
}

/* =============================================================================
   📐 LAYOUT & CONTAINER
   ============================================================================= */

.main {
    background: var(--color-bg-main);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: stretch;
}

.block-container {
    padding: var(--spacing-xl) var(--spacing-md);
    max-width: 900px;
    margin: 0 auto;
    width: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

/* Responsive Container */
@media (max-width: 768px) {
    .block-container {
        padding: var(--spacing-md) var(--spacing-sm);
        max-width: 100%;
    }
}

/* =============================================================================
   🧭 SIDEBAR NAVIGATION
   ============================================================================= */

[data-testid="stSidebar"] {
    background: var(--color-bg-secondary);
    border-right: 1px solid var(--color-border);
    padding-top: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 0;
    display: flex;
    flex-direction: column;
    height: 100%;
}

/* Sidebar Header */
.sidebar-header {
    padding: var(--spacing-lg);
    border-bottom: 1px solid var(--color-border);
    margin-bottom: var(--spacing-sm);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
}

.sidebar-header h2 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--color-text-primary);
    margin: 0;
    letter-spacing: -0.01em;
}

.sidebar-header p {
    font-size: 0.85rem;
    color: var(--color-text-muted);
    margin: var(--spacing-xs) 0 0 0;
}

/* History Items */
.history-item {
    padding: 0.75rem var(--spacing-lg);
    margin: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all var(--transition-base);
    color: var(--color-text-secondary);
    font-size: 0.9rem;
    border-left: 2px solid transparent;
    display: flex;
    align-items: center;
    justify-content: space-between;
    text-align: left;
    word-break: break-word;
}

.history-item:hover {
    background: var(--color-bg-hover);
    border-left-color: var(--color-accent);
    color: var(--color-text-primary);
    transform: translateX(2px);
}

.history-item-active {
    background: var(--color-accent-light);
    border-left-color: var(--color-accent);
    color: var(--color-accent);
}

/* =============================================================================
   🏷️ HEADERS & TITLES
   ============================================================================= */

.main-header {
    text-align: center;
    padding: var(--spacing-2xl) var(--spacing-xl) var(--spacing-xl);
    margin-bottom: var(--spacing-xl);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.main-header h1 {
    font-size: clamp(1.75rem, 4vw, 2.25rem);
    font-weight: 700;
    color: var(--color-text-primary);
    margin: 0 0 var(--spacing-md) 0;
    letter-spacing: -0.02em;
    line-height: var(--line-height-tight);
}

.main-header p {
    font-size: 1rem;
    color: var(--color-text-muted);
    margin: 0;
    max-width: 600px;
    line-height: var(--line-height-normal);
}

/* Section Headers */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--color-text-primary);
    margin: var(--spacing-xl) 0 var(--spacing-md);
    display: flex;
    align-items: center;
    justify-content: flex-start;
}

/* =============================================================================
   📁 FILE UPLOAD AREA
   ============================================================================= */

.upload-container {
    background: var(--color-bg-tertiary);
    border: 2px dashed var(--color-border);
    border-radius: var(--radius-xl);
    padding: var(--spacing-2xl) var(--spacing-xl);
    text-align: center;
    margin-bottom: var(--spacing-xl);
    transition: all var(--transition-base);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 200px;
}

.upload-container:hover {
    border-color: var(--color-accent);
    background: var(--color-bg-hover);
    box-shadow: var(--shadow-accent);
    transform: translateY(-2px);
}

.upload-icon {
    font-size: 3rem;
    margin-bottom: var(--spacing-md);
    color: var(--color-accent);
    display: flex;
    justify-content: center;
    align-items: center;
}

.upload-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--color-text-primary);
    margin-bottom: var(--spacing-sm);
}

.upload-subtitle {
    font-size: 0.95rem;
    color: var(--color-text-secondary);
    margin-bottom: var(--spacing-lg);
    line-height: var(--line-height-normal);
}

/* File Badge */
.file-badge {
    background: var(--color-accent-light);
    color: var(--color-accent);
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-full);
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin: var(--spacing-xs);
}

/* =============================================================================
   💬 CHAT INPUT AREA — FIXED LAYOUT + RESPONSIVE BUTTONS
   ============================================================================= */

.chat-input-container {
    background: var(--color-bg-tertiary);
    border-radius: var(--radius-xl);
    border: 1px solid var(--color-border);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-md);
    margin-bottom: var(--spacing-xl);
    transition: all var(--transition-base);
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: stretch;
    gap: var(--spacing-md);
}

.chat-input-container:focus-within {
    border-color: var(--color-accent);
    box-shadow: var(--shadow-accent);
}

/* Text Area Styling */
.stTextArea textarea {
    background: var(--color-bg-hover) !important;
    border: 1px solid var(--color-border) !important;
    color: var(--color-text-primary) !important;
    font-size: 1rem !important;
    padding: var(--spacing-sm) var(--spacing-md) !important;
    border-radius: var(--radius-md) !important;
    min-height: 90px !important;
    max-height: 250px !important;
    resize: vertical !important;
    line-height: var(--line-height-normal) !important;
}

.stTextArea textarea::placeholder {
    color: var(--color-text-muted) !important;
}

/* Chat Button Row */
.chat-button-row {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    gap: var(--spacing-md) !important;
    flex-wrap: wrap !important;
}

/* =============================================================================
   🔘 BUTTONS — POLISHED INTERACTION & SPACING
   ============================================================================= */

/* Base Button Container */
.stButton,
.secondary-button,
.tertiary-button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}

/* Primary Button (Send) */
.stButton > button {
    background: var(--color-accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1.75rem !important;
    min-width: 130px !important;
    height: 44px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all var(--transition-base) !important;
    box-shadow: var(--shadow-accent) !important;
    white-space: nowrap !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background: var(--color-accent-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 8px rgba(99, 102, 241, 0.45) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Secondary Button (Clear) */
.secondary-button > button {
    background: var(--color-bg-secondary) !important;
    color: var(--color-text-secondary) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.7rem 1.5rem !important;
    min-width: 130px !important;
    height: 44px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: all var(--transition-base) !important;
    cursor: pointer !important;
}

.secondary-button > button:hover {
    background: var(--color-bg-hover) !important;
    border-color: var(--color-accent) !important;
    color: var(--color-accent) !important;
    transform: translateY(-1px) !important;
}

.secondary-button > button:active {
    transform: translateY(0) !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .chat-button-row {
        flex-direction: column !important;
        align-items: stretch !important;
    }

    .stButton > button,
    .secondary-button > button {
        width: 100% !important;
        min-width: unset !important;
    }
}

/* =============================================================================
   🗨️ MESSAGE COMPONENTS
   ============================================================================= */

.message-container {
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg) var(--spacing-lg);
    margin-bottom: var(--spacing-lg);
    box-shadow: var(--shadow-sm);
    transition: all var(--transition-base);
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: stretch;
}

.message-container:hover {
    border-color: var(--color-border-light);
    box-shadow: var(--shadow-md);
}

/* User Message */
.user-message {
    background: var(--color-bg-hover);
    border-left: 3px solid var(--color-accent);
}

/* AI Message */
.ai-message {
    background: var(--color-bg-tertiary);
    border-left: 3px solid var(--color-text-muted);
}

/* Message Header */
.message-header {
    font-weight: 600;
    color: var(--color-text-secondary);
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Message Content */
.message-content {
    color: var(--color-text-primary);
    line-height: var(--line-height-normal);
    font-size: 0.96rem;
    word-wrap: break-word;
    overflow-wrap: break-word;
    text-align: left;
}

.message-content p {
    margin: 0 0 var(--spacing-md) 0;
}

.message-content p:last-child {
    margin-bottom: 0;
}

/* Message Timestamp */
.message-timestamp {
    font-size: 0.8rem;
    color: var(--color-text-muted);
    margin-top: var(--spacing-sm);
    display: flex;
    justify-content: flex-end;
}

/* =============================================================================
   📊 RESULTS & METRICS
   ============================================================================= */

.result-section {
    background: var(--color-bg-tertiary);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
    border: 1px solid var(--color-border);
    margin-bottom: var(--spacing-lg);
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

.result-title {
    font-weight: 600;
    color: var(--color-text-secondary);
    margin-bottom: var(--spacing-md);
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Metrics Container */
div[data-testid="metric-container"] {
    background: var(--color-bg-tertiary) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--spacing-md) !important;
    border: 1px solid var(--color-border) !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: flex-start !important;
}

div[data-testid="metric-container"] label {
    color: var(--color-text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--color-text-primary) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* =============================================================================
   📦 CARDS & GRIDS
   ============================================================================= */

/* =============================================================================
   🏠 WELCOME GRID & CARDS - Spacious and Responsive
   ============================================================================= */

.welcome-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: var(--spacing-xl);
    margin: var(--spacing-2xl) 0;
    padding: 0 var(--spacing-lg);
    justify-items: stretch;
}

.welcome-card {
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-xl);
    padding: var(--spacing-2xl) var(--spacing-xl);
    transition: all var(--transition-base);
    box-shadow: var(--shadow-sm);
    cursor: pointer;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: flex-start;
    text-align: left;
    margin-bottom: 10px        
}

.welcome-card:hover {
    border-color: var(--color-accent);
    box-shadow: var(--shadow-accent);
    transform: translateY(-6px);
}

.card-icon {
    font-size: 2.5rem;
    margin-bottom: var(--spacing-md);
    color: var(--color-accent);
}

.card-title {
    color: var(--color-text-primary);
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: var(--spacing-sm);
}

.card-description {
    color: var(--color-text-secondary);
    font-size: 1rem;
    line-height: var(--line-height-normal);
    margin-top: var(--spacing-sm);
}

/* Extra breathing room on smaller screens */
@media (max-width: 768px) {
    .welcome-grid {
        grid-template-columns: 1fr;
        gap: var(--spacing-lg);
        padding: 0 var(--spacing-md);
    }

    .welcome-card {
        padding: var(--spacing-xl);
    }
}

/* =============================================================================
   🔍 EXAMPLE QUERIES
   ============================================================================= */

.example-query {
    background: var(--color-bg-tertiary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--spacing-md);
    color: var(--color-text-secondary);
    font-size: 0.9rem;
    cursor: pointer;
    transition: all var(--transition-base);
    margin-bottom: var(--spacing-sm);
    display: flex;
    align-items: center;
    justify-content: flex-start;
    text-align: left;
}

.example-query:hover {
    background: var(--color-accent-light);
    border-color: var(--color-accent);
    color: var(--color-accent);
    transform: translateX(4px);
}

/* =============================================================================
   📊 DATA DISPLAY COMPONENTS
   ============================================================================= */

/* DataFrames */
.stDataFrame {
    background: var(--color-bg-tertiary) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--color-bg-tertiary) !important;
    color: var(--color-text-secondary) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--spacing-md) !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--color-accent) !important;
    color: var(--color-accent) !important;
}

/* =============================================================================
   🎨 SEMANTIC ALERTS
   ============================================================================= */

.stSuccess {
    background: var(--color-success-bg) !important;
    border-left: 3px solid var(--color-success-border) !important;
    color: var(--color-success-text) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--spacing-md) !important;
}

.stInfo {
    background: var(--color-info-bg) !important;
    border-left: 3px solid var(--color-info-border) !important;
    color: var(--color-info-text) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--spacing-md) !important;
}

.stError {
    background: var(--color-error-bg) !important;
    border-left: 3px solid var(--color-error-border) !important;
    color: var(--color-error-text) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--spacing-md) !important;
}

.stWarning {
    background: var(--color-warning-bg) !important;
    border-left: 3px solid var(--color-warning-border) !important;
    color: var(--color-warning-text) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--spacing-md) !important;
}

/* =============================================================================
   ⚙️ INTERACTIVE COMPONENTS
   ============================================================================= */

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: var(--spacing-sm);
}

.stTabs [data-baseweb="tab"] {
    background: var(--color-bg-tertiary) !important;
    border: 1px solid var(--color-border) !important;
    color: var(--color-text-muted) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--spacing-sm) var(--spacing-md) !important;
    transition: all var(--transition-base) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--color-bg-hover) !important;
    border-color: var(--color-accent) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--color-accent-light) !important;
    border-color: var(--color-accent) !important;
    color: var(--color-accent) !important;
}

/* Progress Bar */
.stProgress > div > div {
    background: var(--color-accent) !important;
    border-radius: var(--radius-full) !important;
}

.stProgress > div {
    background: var(--color-bg-tertiary) !important;
    border-radius: var(--radius-full) !important;
}

/* Checkbox */
.stCheckbox {
    padding: var(--spacing-sm) 0;
    display: flex;
    align-items: center;
    justify-content: flex-start;
}

.stCheckbox label {
    color: var(--color-text-secondary) !important;
    cursor: pointer;
}

/* Spinner */
.stSpinner > div {
    border-top-color: var(--color-accent) !important;
}

/* =============================================================================
   🧩 UTILITY CLASSES
   ============================================================================= */

/* Divider */
hr {
    border: none;
    border-top: 1px solid var(--color-border);
    margin: var(--spacing-xl) 0;
}

/* Text Alignment */
.text-center {
    text-align: center !important;
}

.text-left {
    text-align: left !important;
}

.text-right {
    text-align: right !important;
}

/* Flex Utilities */
.flex-center {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.flex-between {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
}

.flex-start {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
}

/* Spacing Utilities */
.mt-sm { margin-top: var(--spacing-sm) !important; }
.mt-md { margin-top: var(--spacing-md) !important; }
.mt-lg { margin-top: var(--spacing-lg) !important; }
.mb-sm { margin-bottom: var(--spacing-sm) !important; }
.mb-md { margin-bottom: var(--spacing-md) !important; }
.mb-lg { margin-bottom: var(--spacing-lg) !important; }

/* =============================================================================
   📱 RESPONSIVE DESIGN
   ============================================================================= */

@media (max-width: 768px) {
    .main-header h1 {
        font-size: 1.75rem;
    }
    
    .welcome-grid {
        grid-template-columns: 1fr;
    }
    
    .button-group {
        flex-direction: column;
        align-items: stretch !important;
    }
    
    .stButton > button,
    .secondary-button > button {
        width: 100% !important;
    }
    
    .upload-container {
        padding: var(--spacing-lg);
    }
}

/* =============================================================================
   🎭 ANIMATIONS
   ============================================================================= */

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeIn 0.3s ease-in-out;
}

/* =============================================================================
   🌐 ACCESSIBILITY IMPROVEMENTS
   ============================================================================= */

*:focus-visible {
    outline: 2px solid var(--color-accent);
    outline-offset: 2px;
}

/* High Contrast Focus for Buttons */
button:focus-visible,
a:focus-visible {
    outline: 2px solid var(--color-accent) !important;
    outline-offset: 2px !important;
}

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>

""", unsafe_allow_html=True)

# Initialize session state
if 'loaded_files' not in st.session_state:
    st.session_state.loaded_files = []
if 'loaded_documents' not in st.session_state:
    st.session_state.loaded_documents = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'welcome'
if 'cleaning_options' not in st.session_state:
    st.session_state.cleaning_options = {
        'remove_non_numeric': True,
        'handle_concatenated': True,
        'remove_special_chars': True,
        'convert_negative': True
    }
if 'vector_db_path' not in st.session_state:
    st.session_state.vector_db_path = Path("data/vector_db")
    st.session_state.vector_db_path.mkdir(parents=True, exist_ok=True)

def get_file_hash(df):
    """Generate unique hash for CSV content"""
    content_str = df.to_csv(index=False)
    return hashlib.md5(content_str.encode()).hexdigest()

def get_vector_db_info():
    """Get information about the vector database"""
    try:
        db_path = st.session_state.vector_db_path
        if not db_path.exists():
            return {'exists': False, 'collections': 0, 'size_mb': 0}
        
        size_bytes = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        size_mb = size_bytes / 1024 / 1024
        collections = len([d for d in db_path.iterdir() if d.is_dir()])
        
        return {
            'exists': True,
            'collections': collections,
            'size_mb': size_mb,
            'path': str(db_path)
        }
    except Exception as e:
        return {'exists': False, 'error': str(e)}

def clear_vector_db():
    """Clear the vector database"""
    try:
        import shutil
        db_path = st.session_state.vector_db_path
        if db_path.exists():
            shutil.rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        st.error(f"Error clearing vector DB: {str(e)}")
        return False

def clean_numeric_string(value):
    """Clean malformed numeric strings"""
    if pd.isna(value) or value == '':
        return np.nan
    
    value_str = str(value).strip()
    
    if value_str.endswith('-'):
        value_str = '-' + value_str[:-1]
    
    value_str = re.sub(r'[^\d.\-eE+]', '', value_str)
    
    if value_str.count('.') > 1 or (value_str.count('-') > 1 and not value_str.startswith('-')):
        match = re.search(r'^-?\d+\.?\d*', value_str)
        if match:
            value_str = match.group()
    
    try:
        return float(value_str) if value_str else np.nan
    except (ValueError, TypeError):
        return np.nan

def clean_csv_data(df):
    """Clean CSV data to handle malformed numeric values"""
    df_cleaned = df.copy()
    
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            sample = df_cleaned[col].dropna().head(10)
            if len(sample) > 0:
                numeric_pattern = any(bool(re.search(r'\d', str(val))) for val in sample)
                
                if numeric_pattern:
                    cleaned_col = df_cleaned[col].apply(clean_numeric_string)
                    valid_ratio = cleaned_col.notna().sum() / len(cleaned_col)
                    if valid_ratio > 0.5:
                        df_cleaned[col] = cleaned_col
    
    return df_cleaned

def load_csv_file(uploaded_file, clean_data=True):
    """Load a CSV file and ingest it"""
    try:
        df = pd.read_csv(uploaded_file)
        original_shape = df.shape
        
        if clean_data:
            df = clean_csv_data(df)
        
        file_hash = get_file_hash(df)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        file_id, metadata = csv_ingestion.ingest_csv(tmp_path, vectorize=True)
        
        os.unlink(tmp_path)
        
        return {
            'name': uploaded_file.name,
            'file_id': file_id,
            'file_hash': file_hash,
            'metadata': metadata,
            'status': 'success',
            'original_shape': original_shape,
            'cleaned_shape': df.shape,
            'cleaned': clean_data,
            'type': 'csv'
        }
    except Exception as e:
        return {
            'name': uploaded_file.name,
            'status': 'error',
            'error': str(e),
            'type': 'csv'
        }

def load_document_file(uploaded_file):
    """Load a document file (TXT/DOCX)"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix, mode='wb') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        file_id, metadata = document_ingestion.ingest_document(tmp_path, vectorize=True)
        
        os.unlink(tmp_path)
        
        return {
            'name': uploaded_file.name,
            'file_id': file_id,
            'metadata': metadata,
            'status': 'success',
            'type': 'document'
        }
    except Exception as e:
        return {
            'name': uploaded_file.name,
            'status': 'error',
            'error': str(e),
            'type': 'document'
        }

def display_results(result):
    """Display query results"""
    if 'error' in result:
        st.error(f"❌ {result.get('details', result['error'])}")
        return
    
    metadata = result.get('metadata', {})
    intent = metadata.get('intent', '')
    
    # Document query
    if intent == 'document_query':
        narrative = result.get('narrative', 'No response generated.')
        st.markdown(f'<div class="message-content">{narrative}</div>', unsafe_allow_html=True)
        
        if 'result_table' in result and result['result_table']:
            with st.expander("📑 View detailed information", expanded=False):
                for idx, chunk_data in enumerate(result['result_table'][:3], 1):
                    st.markdown(f"**Section {idx}**")
                    
                    if chunk_data.get('question'):
                        st.info(f"❓ {chunk_data['question']}")
                        st.success(f"✓ {chunk_data['answer']}")
                        if chunk_data.get('analysis'):
                            st.warning(f"📈 {chunk_data['analysis']}")
                    else:
                        st.text(chunk_data.get('content', '')[:500])
                    
                    if idx < 3:
                        st.divider()
        
        numbers = result.get('numbers', {})
        if numbers:
            col1, col2 = st.columns(2)
            with col1:
                if 'num_results' in numbers:
                    st.metric("Results Found", numbers['num_results'])
            with col2:
                if 'avg_similarity' in numbers:
                    st.metric("Relevance", f"{numbers['avg_similarity']:.3f}")
        return
    
    # General query
    if intent == 'general_query':
        narrative = result.get('narrative', 'No response generated.')
        st.markdown(f'<div class="message-content">{narrative}</div>', unsafe_allow_html=True)
        
        numbers = result.get('numbers', {})
        if numbers:
            cols = st.columns(3)
            if 'csv_files' in numbers:
                with cols[0]:
                    st.metric("CSV Files", numbers['csv_files'])
            if 'document_files' in numbers:
                with cols[1]:
                    st.metric("Documents", numbers['document_files'])
            if 'total_rows' in numbers:
                with cols[2]:
                    st.metric("Total Rows", numbers['total_rows'])
        return
    
    # Analytical queries
    numbers = result.get('numbers', {})
    
    if 'narrative' in result and result['narrative']:
        st.info(f"💡 {result['narrative']}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="result-title">📊 Results</div>', unsafe_allow_html=True)
        
        if 'top_values' in numbers:
            values = numbers['top_values']
            indices = numbers.get('top_indices', [])
            
            if isinstance(values, list):
                results_df = pd.DataFrame({
                    'Value': values,
                    'Row Index': indices
                })
                st.dataframe(results_df, use_container_width=True)
            else:
                st.metric("Result", values)
        
        if 'result_table' in result and result['result_table']:
            result_df = pd.DataFrame(result['result_table'])
            st.dataframe(result_df, use_container_width=True)
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="results.csv",
                mime="text/csv",
                key=f"download_{uuid.uuid4()}"
            )
    
    with col2:
        st.markdown('<div class="result-title">📈 Statistics</div>', unsafe_allow_html=True)
        
        stats = [
            ('average', 'Average', '📊'),
            ('min', 'Minimum', '⬇️'),
            ('max', 'Maximum', '⬆️'),
            ('sum', 'Total', '➕'),
            ('count', 'Count', '🔢'),
        ]
        
        for key, label, emoji in stats:
            if key in numbers:
                value = numbers[key]
                if isinstance(value, float):
                    st.metric(f"{emoji} {label}", f"{value:.2f}")
                else:
                    st.metric(f"{emoji} {label}", value)

def main():
    # Sidebar - History and Settings
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h2>💬 Conversations</h2></div>', unsafe_allow_html=True)
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.query_history = []
            st.session_state.current_view = 'welcome'
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # History
        if st.session_state.query_history:
            st.markdown("### Recent")
            for idx, item in enumerate(st.session_state.query_history):
                query_preview = item['query'][:40] + "..." if len(item['query']) > 40 else item['query']
                if st.button(query_preview, key=f"history_{idx}", use_container_width=True):
                    st.session_state.current_view = 'chat'
                    st.rerun()
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings", expanded=False):
            st.markdown("**Data Cleaning**")
            st.session_state.cleaning_options['remove_non_numeric'] = st.checkbox(
                "Clean numeric data", value=True
            )
            st.session_state.cleaning_options['handle_concatenated'] = st.checkbox(
                "Fix merged numbers", value=True
            )
            st.session_state.cleaning_options['convert_negative'] = st.checkbox(
                "Fix negative signs", value=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Database info
            db_info = get_vector_db_info()
            if db_info['exists']:
                st.markdown("**Database**")
                st.caption(f"Size: {db_info['size_mb']:.1f} MB")
                if st.button("Clear Database", use_container_width=True, key="clear_db"):
                    if clear_vector_db():
                        st.success("Database cleared")
                        st.rerun()
        
        st.divider()
        
        # Loaded files
        if st.session_state.loaded_files or st.session_state.loaded_documents:
            st.markdown("### 📁 Loaded Files")
            
            for file_info in st.session_state.loaded_files:
                if file_info['status'] == 'success':
                    st.markdown(f'<div class="file-badge">📊 {file_info["name"]}</div>', unsafe_allow_html=True)
            
            for doc_info in st.session_state.loaded_documents:
                if doc_info['status'] == 'success':
                    st.markdown(f'<div class="file-badge">📄 {doc_info["name"]}</div>', unsafe_allow_html=True)
            
            if st.button("Clear All Files", use_container_width=True, key="clear_files"):
                st.session_state.loaded_files = []
                st.session_state.loaded_documents = []
                st.rerun()
    
    # Main content
    if not st.session_state.loaded_files and not st.session_state.loaded_documents:
        # Welcome screen with upload
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Trust First AI Analyst</h1>
            <p>Upload your files and start asking questions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload section
        st.markdown("""
        <div class="upload-container">
            <div class="upload-icon">📁</div>
            <div class="upload-title">Upload Your Files</div>
            <div class="upload-subtitle">Support for CSV files and documents (TXT, DOCX)</div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["CSV Files", "Documents"])
        
        with tab1:
            uploaded_csvs = st.file_uploader(
                "Choose CSV files",
                type=['csv'],
                accept_multiple_files=True,
                key="csv_upload"
            )
            
            if uploaded_csvs:
                if st.button("Load CSV Files", type="primary", use_container_width=True):
                    st.session_state.loaded_files = []
                    progress = st.progress(0)
                    
                    clean_enabled = any(st.session_state.cleaning_options.values())
                    
                    for idx, file in enumerate(uploaded_csvs):
                        file.seek(0)
                        result = load_csv_file(file, clean_data=clean_enabled)
                        st.session_state.loaded_files.append(result)
                        progress.progress((idx + 1) / len(uploaded_csvs))
                    
                    success = sum(1 for f in st.session_state.loaded_files if f['status'] == 'success')
                    if success > 0:
                        st.success(f"✓ Loaded {success} file(s)")
                        st.session_state.current_view = 'chat'
                        st.rerun()
        
        with tab2:
            uploaded_docs = st.file_uploader(
                "Choose document files",
                type=['txt', 'docx'],
                accept_multiple_files=True,
                key="doc_upload"
            )
            
            if uploaded_docs:
                if st.button("Load Documents", type="primary", use_container_width=True):
                    st.session_state.loaded_documents = []
                    progress = st.progress(0)
                    
                    for idx, file in enumerate(uploaded_docs):
                        file.seek(0)
                        result = load_document_file(file)
                        st.session_state.loaded_documents.append(result)
                        progress.progress((idx + 1) / len(uploaded_docs))
                    
                    success = sum(1 for f in st.session_state.loaded_documents if f['status'] == 'success')
                    if success > 0:
                        st.success(f"✓ Loaded {success} document(s)")
                        st.session_state.current_view = 'chat'
                        st.rerun()
        
        # Welcome cards
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### What can I help you with?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="welcome-card">
                <div class="card-icon"></div>
                <div class="card-title">Analyze CSV Data</div>
                <div class="card-description">
                    Upload CSV files and ask questions about your data. Get insights, statistics, and visualizations.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="welcome-card">
                <div class="card-icon"></div>
                <div class="card-title">Find Insights</div>
                <div class="card-description">
                    Discover patterns, trends, and outliers in your data with natural language queries.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="welcome-card">
                <div class="card-icon"></div>
                <div class="card-title">Search Documents</div>
                <div class="card-description">
                    Upload documents and get answers from your content. Perfect for research and analysis.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="welcome-card">
                <div class="card-icon"></div>
                <div class="card-title">Quick Results</div>
                <div class="card-description">
                    Get instant answers with AI-powered analysis. No complex queries or coding required.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Chat interface with loaded files
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Trust First AI Analyst</h1>
            <p>Ask me anything about your data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display conversation history
        if st.session_state.query_history:
            for idx, item in enumerate(st.session_state.query_history):
                # User message
                st.markdown(f"""
                <div class="message-container user-message">
                    <div class="message-header">You</div>
                    <div class="message-content">{item['query']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # AI response
                st.markdown("""
                <div class="message-container">
                    <div class="message-header">🤖 AI Analyst</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.container():
                    display_results(item['result'])
        else:
            # Show example queries if no history
            st.markdown("### 💡 Example Questions")
            
            examples = [
                "What is the average of all values?",
                "Show me the top 5 highest values",
                "What is the minimum and maximum?",
                "Summarize the main points from my document",
                "How many rows are in the dataset?"
            ]
            
            for example in examples:
                st.markdown(f'<div class="example-query">💬 {example}</div>', unsafe_allow_html=True)
        
        # Input area at the bottom
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        
        query = st.text_area(
            "Message AI Analyst...",
            height=100,
            placeholder="Type your question here...",
            label_visibility="collapsed",
            key="query_input"
        )
        
        # Buttons Row (Send + Clear)
        st.markdown('<div class="chat-button-row">', unsafe_allow_html=True)

        send_button = st.button("Send", type="primary")
        st.markdown('<div class="secondary-button">', unsafe_allow_html=True)
        clear_button = st.button("Clear")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle query submission
        if send_button and query:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = analytical_agent.process_query(query)
                    
                    st.session_state.query_history.append({
                        'query': query,
                        'result': result
                    })
                    
                    # Keep only last 10 conversations
                    st.session_state.query_history = st.session_state.query_history[-10:]
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if clear_button:
            st.session_state.query_history = []
            st.rerun()

if __name__ == "__main__":
    main()