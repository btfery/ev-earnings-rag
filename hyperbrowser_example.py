import os
from dotenv import load_dotenv
from hyperbrowser import Hyperbrowser
from hyperbrowser.models import FetchParams, FetchOutputOptions, FetchBrowserOptions, FetchNavigationOptions


load_dotenv()

client = Hyperbrowser(api_key=os.getenv("HYPERBROWSER_API_KEY"))


def main():
    fetch_result = client.web.fetch(params=FetchParams(
      url="https://www.fool.com/earnings/call-transcripts/2026/02/13/rivian-rivn-q4-2025-earnings-call-transcript/",
      stealth="auto",
      outputs=FetchOutputOptions(
        formats=[
          "markdown",
          {
            "type": "json",
            "prompt": "Only include the text from the \"Full Conference Call Transcript\" section"
          }
        ],
        sanitize="basic",
        include_selectors=[],
        exclude_selectors=[]
      ),
      browser=FetchBrowserOptions(
        screen={
          "width": 1280,
          "height": 720
        }
      ),
      navigation=FetchNavigationOptions(
        wait_until="domcontentloaded"
      )
    ))
    print("Fetch result:\n", fetch_result)


if __name__ == "__main__":
    main()