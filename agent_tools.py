import json
import urllib.request
import urllib.parse
from answer_extractor import safe_eval_expression

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Returns the result as a string, or an error message if invalid.
    """
    try:
        # Evaluate using the robust SymPy extractor
        result = safe_eval_expression(expression)
        
        # Format result back to string
        if result is None:
            return "Error: Could not parse or evaluate expression."
        
        return str(result)
        
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def math_cheatsheet(topic: str) -> str:
    """Retrieve math concepts/formulas using a simple 'RAG' approach via Wikipedia API.
    Looks up the topic on Wikipedia and returns the summary/extract.
    """
    try:
        # Search for the closest matching Wikipedia page
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(topic)}&utf8=&format=json"
        
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            search_data = json.loads(response.read())
            
        search_results = search_data.get('query', {}).get('search', [])
        
        if not search_results:
            return f"No mathematical concepts found for topic: {topic}"
            
        # Get the title of the top hit
        top_title = search_results[0]['title']
        
        # Fetch the extract (summary) of the top page
        fetch_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exsentences=5&exlimit=1&titles={urllib.parse.quote(top_title)}&explaintext=1&format=json"
        req = urllib.request.Request(fetch_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            page_data = json.loads(response.read())
            
        pages = page_data.get('query', {}).get('pages', {})
        
        for page_id, page_info in pages.items():
            if 'extract' in page_info:
                # Truncate to avoid exploding context windows (e.g. max 1000 characters)
                extract = page_info['extract'].strip()
                if len(extract) > 1000:
                    extract = extract[:997] + "..."
                return f"Information about '{top_title}':\n{extract}"
                
        return f"Could not retrieve content for topic: {topic}"
        
    except Exception as e:
        return f"Error retrieving information: {str(e)}"

# Test the tools when run directly
if __name__ == "__main__":
    print("Testing calculator('45**2'):")
    print(calculator("45**2"))
    print("\nTesting calculator('integrate(x**2, (x, 0, 1))'):")
    print(calculator("integrate(x**2, (x, 0, 1))"))
    print("\nTesting math_cheatsheet('Pythagorean theorem'):")
    print(math_cheatsheet("Pythagorean theorem"))
