# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# import json
# import re

# sns.set_theme(style="whitegrid", context="notebook")

# # ---------------------------------------------------------
# # 1. SETUP & DATA LOADING
# # ---------------------------------------------------------
# def load_data(json_data):
#     """
#     Loads JSON data into a DataFrame.
#     """
#     df = pd.DataFrame(json_data)
#     print(f"✅ Data Loaded. Total Chunks: {len(df)}")
#     return df

# def resolve_text_column(df: pd.DataFrame) -> str:
#     """Find the best available text column in the dataset."""
#     candidates = ["page_content", "text", "content", "body", "raw_text", "markdowntext"]
#     for c in candidates:
#         if c in df.columns:
#             return c
#     raise KeyError("No text column found. Expected one of: " + ", ".join(candidates))

# def normalize_text(s: str) -> str:
#     """Basic text normalization to improve analysis quality."""
#     s = s.lower()
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def clean_data(df: pd.DataFrame, min_words: int = 10, max_words: int = 1200) -> pd.DataFrame:
#     """
#     Clean and standardize data for quality analysis.
#     - Removes empty/null text
#     - Normalizes whitespace/casing
#     - Filters out extremely short/long chunks (noise or merged pages)
#     """
#     text_col = resolve_text_column(df)
#     df = df.copy()
#     df[text_col] = df[text_col].astype(str).fillna("")
#     df[text_col] = df[text_col].str.replace(r"\s+", " ", regex=True).str.strip()

#     # Remove empty rows
#     df = df[df[text_col] != ""]

#     # Word/char counts + remove very short/very long chunks
#     df["word_count"] = df[text_col].apply(lambda x: len(x.split()))
#     df["char_count"] = df[text_col].apply(lambda x: len(x))
#     df = df[(df["word_count"] >= min_words) & (df["word_count"] <= max_words)]

#     print(f"✅ Remaining rows after cleaning: {len(df)}")
#     return df, text_col

# # ---------------------------------------------------------
# # 2. ANALYSIS (NO DEDUP REMOVAL FOR RAG CHUNKS)
# # ---------------------------------------------------------
# def analyze_completeness(df, text_col: str):
#     """
#     Checks for missing data and content length.
#     """
#     print("\n--- 1. DATA COMPLETENESS ---")
#     null_count = df[text_col].isnull().sum()
#     empty_count = df[df[text_col].str.strip() == ""].shape[0]
#     total_issues = null_count + empty_count
#     avg_words = df["word_count"].mean()
#     avg_chars = df["char_count"].mean()
#     short_chunks = df[df["word_count"] < 10].shape[0]
#     print(f"• Empty/Null Rows: {total_issues}")
#     print(f"• Average Word Count per Chunk: {avg_words:.0f}")
#     print(f"• Average Character Count per Chunk: {avg_chars:.0f}")
#     print(f"• 'Noise' Chunks (<10 words): {short_chunks} ({(short_chunks/len(df))*100:.1f}%)")
#     return df

# # ---------------------------------------------------------
# # 3. VISUALIZATION (SEPARATE FIGURES)
# # ---------------------------------------------------------
# def plot_completeness_distributions(df, mode: str = "overlay"):
#     """Plot word and character distributions either overlayed or separate."""
#     if mode == "overlay":
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.histplot(df["word_count"], bins=40, kde=True, color="steelblue", ax=ax, label="Words", alpha=0.5)
#         sns.histplot(df["char_count"], bins=40, kde=True, color="teal", ax=ax, label="Chars", alpha=0.5)
#         ax.set_title("Word vs Character Distribution")
#         ax.set_xlabel("Count")
#         ax.set_ylabel("Frequency")
#         ax.set_yscale("log")
#         ax.legend()
#         plt.tight_layout()
#         plt.show()
#         return fig

#     fig, axes = plt.subplots(2, 1, figsize=(8, 9))
#     sns.histplot(df["word_count"], bins=40, kde=True, color="steelblue", ax=axes[0])
#     axes[0].set_title("Word Count Distribution")
#     axes[0].set_xlabel("Words per Chunk")
#     axes[0].set_ylabel("Frequency")
#     axes[0].set_yscale("log")
#     axes[0].axvline(df["word_count"].mean(), color="red", linestyle="--", label=f"Avg: {df['word_count'].mean():.0f}")
#     axes[0].legend()

#     sns.histplot(df["char_count"], bins=40, kde=True, color="teal", ax=axes[1])
#     axes[1].set_title("Character Count Distribution")
#     axes[1].set_xlabel("Characters per Chunk")
#     axes[1].set_ylabel("Frequency")
#     axes[1].set_yscale("log")
#     axes[1].axvline(df["char_count"].mean(), color="red", linestyle="--", label=f"Avg: {df['char_count'].mean():.0f}")
#     axes[1].legend()

#     plt.tight_layout()
#     plt.show()
#     return fig

# def plot_top_sources(df):
#     fig, ax = plt.subplots(figsize=(8, 5))
#     if "metadata" in df.columns:
#         sources = df["metadata"].apply(lambda m: m.get("source") if isinstance(m, dict) else "unknown")
#         top_sources = sources.value_counts().head(10)

#         sns.barplot(x=top_sources.values, y=top_sources.index, hue=top_sources.index, legend=False, ax=ax, palette="viridis")
#         ax.set_title("Top Sources")
#         ax.set_xlabel("Count")
#         ax.set_ylabel("Source")

#         # Dynamic font size: bigger for frequent, smaller for less frequent
#         counts = top_sources.values
#         min_c, max_c = counts.min(), counts.max()
#         min_fs, max_fs = 8, 14
#         if max_c == min_c:
#             sizes = [max_fs] * len(counts)
#         else:
#             sizes = [min_fs + (c - min_c) * (max_fs - min_fs) / (max_c - min_c) for c in counts]
#         ax.set_yticklabels(top_sources.index)
#         for label, fs in zip(ax.get_yticklabels(), sizes):
#             label.set_fontsize(fs)
#     else:
#         ax.axis("off")
#         ax.text(0.5, 0.5, "No source metadata", ha="center", va="center")

#     plt.tight_layout()
#     plt.show()
#     return fig

# # ---------------------------------------------------------
# # 4. EXECUTION
# # ---------------------------------------------------------
# try:
#     data
# except NameError:
#     with open("extracted_text_data/checkpoints/06_final_combined_20260209_172552.json", "r") as file:
#         data = json.load(file)

#     # with open("extracted_text_data/checkpoints/06_final_combined_20260209_101614.json", "r") as file:
#     #     data2 = json.load(file)

# df_raw = load_data(data)
# df_clean, text_col = clean_data(df_raw, min_words=10, max_words=1200)
# df_clean = analyze_completeness(df_clean, text_col)
# plot_completeness_distributions(df_clean, mode="separate")
# plot_top_sources(df_clean)





