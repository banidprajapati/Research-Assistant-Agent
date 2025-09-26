from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import feedparser

# ----------- Step 1: Load the VSLM -----------
# Using FLAN-T5-base which is better trained for instruction following
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# ----------- Step 2: Function to rewrite query -----------
def rewrite_query(user_query):
    # Few-shot prompting for better results
    few_shot_prompt = """Rewrite the following user queries into concise academic search keywords suitable for arXiv.

Example 1:
Input: beginner friendly research papers on machine learning
Output: introductory machine learning, ML survey, ML tutorial

Example 2:
Input: AI papers for kids
Output: introductory AI, AI tutorial, AI overview

Now rewrite the next query:
Input: {}
Output:""".format(user_query)
    
    input_ids = tokenizer(few_shot_prompt, return_tensors="pt", max_length=512, truncation=True).input_ids

    # Generate with parameters optimized for FLAN-T5
    outputs = model.generate(
        input_ids, 
        max_length=80, 
        min_length=5,
        num_beams=3, 
        early_stopping=True,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    rewritten_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return rewritten_query

# ----------- Step 3: Function to fetch papers from arXiv -----------
def fetch_top_arxiv_papers(query, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    feed = feedparser.parse(url)
    papers = []
    for entry in feed.entries:
        paper = {
            "title": entry.title,
            "authors": [author.name for author in entry.authors],
            "summary": entry.summary,
            "pdf_link": entry.link.replace("abs", "pdf")
        }
        papers.append(paper)
    return papers

# ----------- Step 4: Example usage -----------
user_query = "top papers for implementation of RAG system"
rewritten_query = rewrite_query(user_query)
print("Initial Query:", rewritten_query)
rewritten_query = "+".join([kw.strip() for kw in rewritten_query.split(",")])
print("Rewritten Query:", rewritten_query)

top_papers = fetch_top_arxiv_papers(rewritten_query)

for idx, paper in enumerate(top_papers, start=1):
    print(f"\n{idx}. {paper['title']}")
    print(f"   Authors: {', '.join(paper['authors'])}")
    print(f"   PDF: {paper['pdf_link']}")
    print(f"   Summary: {paper['summary']}")
