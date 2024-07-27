"""
generate Arxiv dataset and preprocess it to generate pairs
"""

import arxiv
from multiprocessing import Pool
import pandas as pd

nresults = 1000
df = pd.read_csv("../../data/arxiv/categories.txt", header=None, delimiter="\t")
categories = df.iloc[:,0].to_list()

"""
input: qurery keyword to the arxiv API
    writes to a file entries of the format <paper title><tab><category>
"""
def fetch_title_and_categories_from_keyword(keyword):
    client = arxiv.Client()
    search = arxiv.Search(query = keyword, 
                          max_results = nresults,
                          sort_by=arxiv.SortCriterion.SubmittedDate)
    
    results = client.results(search)
    category_sets = {} 

    with open(f"../../data/arxiv/data_{keyword}.taxo", "w") as f:
        print(f"writing to data_{keyword}.taxo...")
        for res in results:
            for category in res.categories:
                if category not in categories:
                    continue # this is to only include valid categories
                f.write(f"{res.title}\t{category}\n")
                if category not in category_sets.keys():
                    category_sets[category] = [res.title]
                else:
                    category_sets[category].append(res.title)
"""
input: list of keywords, multiprocess to query from arxiv
"""
def fetch_title_and_categories(keywords, ncpus=5):
    with Pool(ncpus) as p:
        p.map(fetch_title_and_categories_from_keyword, keywords)
        
if __name__ == "__main__":
    fetch_title_and_categories(["algorithm", "physics", "operator", "financial", "volatility"])