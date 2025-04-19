# town_meetings

Given a town name, extract where and when they meet and where to find their meetings and agendas.

This project uses the Anthropic Python SDK as well as a custom BeautifulSoup tool that handles the website parsing and filtering.

To get setup:

```
# Create a virtual environment
uv venv

# Install dependencies
uv sync

# Activate virtual env
source .venv/bin/activate
```

Currently, it is setup as a basic script with the municipality name hard coded in `read_website.py`. Usage: `python read_website.py`
