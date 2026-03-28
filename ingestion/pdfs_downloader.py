import os
import arxiv

query = 'ti:"transformer" OR abs:"transformer" AND (cat:cs.LG OR cat:cs.CL)'

search = arxiv.Search(
    query=query,
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

client = arxiv.Client()
papers = list(client.results(search))

# create nested folder data/raw_papers
output_dir = os.path.join("data", "raw_papers")
os.makedirs(output_dir, exist_ok=True)

for paper in papers:
    paper.download_pdf(dirpath=output_dir)