import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ================================
# LOAD DATA (IMPORTANT)
# ================================

df = pd.read_csv("sample_data.csv")   # ✅ Make sure this file is in GitHub

# ================================
# JOB DESCRIPTIONS
# ================================

job_descriptions = [
    {"role": "Data Analyst", "description": "Python SQL Excel data analysis"},
    {"role": "HR Analyst", "description": "HR recruitment employee management"},
]

jd_df = pd.DataFrame(job_descriptions)

# ================================
# CLEAN DATA
# ================================

df['Lemmatized_text'] = df['Lemmatized_text'].fillna("").astype(str)

# Convert string list → actual list
df['Keywords'] = df['Keywords'].apply(
    lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else []
)

jd_df['description'] = jd_df['description'].fillna("").astype(str)

# ================================
# EMBEDDINGS
# ================================

model = SentenceTransformer('all-MiniLM-L6-v2')

resume_embeddings = model.encode(df['Lemmatized_text'].tolist())
jd_embeddings = model.encode(jd_df['description'].tolist())

all_similarity_scores = cosine_similarity(jd_embeddings, resume_embeddings)

# ================================
# UI
# ================================

st.title("🚀 GenAI-Powered Intelligent Hiring Assistant")

selected_role = st.selectbox("Select Job Role", jd_df["role"])

# ================================
# FUNCTIONS
# ================================

def skill_match_score(jd_keywords, resume_keywords):
    jd_set = set(jd_keywords)
    resume_set = set(resume_keywords)
    
    return len(jd_set.intersection(resume_set)) / len(jd_set) if len(jd_set) > 0 else 0


def experience_score(text):
    wc = len(text.split())
    
    if wc < 100:
        return 0.3
    elif wc < 300:
        return 0.6
    else:
        return 1.0


def final_score(sim, skill, exp):
    return 0.4 * sim + 0.5 * skill + 0.1 * exp


def rank_candidates(jd_index, top_n=5):
    results = []
    role = jd_df.iloc[jd_index]['role']
    
    for i in range(len(df)):
        resume_keywords = df.iloc[i]['Keywords']
        
        # 🔥 FILTER (important for relevance)
        if role == "Data Analyst":
            required = ["python", "sql", "data"]
            if sum(1 for s in required if s in resume_keywords) < 1:
                continue
        
        if role == "HR Analyst":
            required = ["hr", "recruitment"]
            if sum(1 for s in required if s in resume_keywords) < 1:
                continue
        
        # SCORES
        sim = all_similarity_scores[jd_index][i]
        
        skill = skill_match_score(
            jd_df.iloc[jd_index]['description'].split(),
            resume_keywords
        )
        
        if skill < 0.05:
            continue
        
        exp = experience_score(df.iloc[i]['Lemmatized_text'])
        
        score = final_score(sim, skill, exp)
        
        results.append((i, score))
    
    results = sorted(results, key=lambda x: x[1], reverse=True)
    
    return results[:top_n]


def generate_explanation(idx, role):
    skills = df.iloc[idx]['Keywords']
    
    # ✅ Remove duplicates
    unique_skills = list(dict.fromkeys(skills))
    
    if role == "Data Analyst":
        req = ["python", "sql", "excel", "data"]
    elif role == "HR Analyst":
        req = ["hr", "recruitment", "employee"]
    else:
        req = []
    
    matched = [s for s in unique_skills if s in req][:3]
    
    if len(matched) > 0:
        return f"Strong skills in {', '.join(matched)} relevant to {role} role."
    else:
        return f"Relevant experience aligned with {role} requirements."

# ================================
# BUTTON ACTION
# ================================

if st.button("Find Best Candidates"):
    
    jd_index = jd_df[jd_df["role"] == selected_role].index[0]
    
    results = rank_candidates(jd_index)
    
    if len(results) > 0:
        
        st.subheader("🏆 Top Candidates")
        st.success(f"Found {len(results)} matching candidates")
        
        for idx, score in results:
            
            explanation = generate_explanation(idx, selected_role)
            
            st.markdown(f"""
            ---
            **Candidate ID:** {idx}  
            **Score:** {round(float(score),2)}  
            **Explanation:** {explanation}
            """)
    
    else:
        st.warning("No candidates found")
