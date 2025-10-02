from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import feedparser
import json

# ----------- Step 1: Load the VSLM -----------
# Using FLAN-T5-base which is better trained for instruction following
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


# ----------- Step 2: Function to rewrite query -----------
def rewrite_query(user_query):
    # Few-shot prompting for better results
    few_shot_prompt = f"""Rewrite the following user queries into concise academic search keywords suitable for arXiv.

Example 1:
Input: beginner friendly research papers on machine learning
Output: introductory machine learning, ML survey, ML tutorial

Example 2:
Input: AI papers for kids
Output: introductory AI, AI tutorial, AI overview

Now rewrite the next query:
Input: {user_query}
Output:"""
    input_ids = tokenizer(
        few_shot_prompt, return_tensors="pt", max_length=512, truncation=True
    ).input_ids

    # Generate with parameters optimized for FLAN-T5
    outputs = model.generate(
        input_ids,
        max_length=80,
        min_length=5,
        num_beams=3,
        early_stopping=True,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    rewritten_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return rewritten_query


# ----------- Step 3: Function to fetch papers from arXiv -----------
def fetch_top_arxiv_papers(query, max_results=10):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    feed = feedparser.parse(url)
    papers = []
    for entry in feed.entries:
        paper_info = {
            "title": entry.title,
            "authors": [author.name for author in entry.authors],
            "summary": entry.summary,
            "pdf_link": entry.link.replace("abs", "pdf"),
        }
        papers.append(paper_info)
    return papers


# ----------- Step 4: Save Research Session -----------
def save_research_session(user_query, rewritten_query, papers):
    session = {
        "original_query": user_query,
        "rewritten_query": rewritten_query,
        "papers": papers,
    }
    with open(f"Output/{user_query}.json", "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)
    print(f"Research session saved to Outout/{user_query}.json")


# ----------- Step 5: Usage -----------
user_query = "Nepal"
rewritten_query = rewrite_query(user_query)
rewritten_query = "+".join(
    [kw.strip().replace(" ", "+") for kw in rewritten_query.split(",")]
)

papers = fetch_top_arxiv_papers(rewritten_query)

# Save the research session
save_research_session(user_query, rewritten_query, papers)
