from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from bs4 import BeautifulSoup
from .webpage.webpage_resources import meta_tags_html

class MetaTagsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip static assets, service worker, favicon, etc.
        if request.url.path.startswith(("/assets", "/pwa", "/favicon.ico")):
            return await call_next(request)
        response: Response = await call_next(request)

        if request.url.path == "/" and response.headers.get("content-type", "").startswith("text/html"):
            body = b"".join([chunk async for chunk in response.body_iterator])
            soup = BeautifulSoup(body, "html.parser")

            # Remove any existing meta tags
            for tag in soup.head.find_all("meta"):
                tag.decompose()

            # Parse your saved meta tags and append them one by one
            meta_soup = BeautifulSoup(meta_tags_html, "html.parser")
            for tag in meta_soup.find_all(["meta", "title"]):  # include <title> if you keep it there
                soup.head.append(tag)

            modified_content = str(soup).encode("utf-8")
            headers = dict(response.headers)
            headers["content-length"] = str(len(modified_content))

            return Response(
                content=modified_content,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type
            )

        return response
