import asyncio, sys
from playwright.async_api import async_playwright
SP = sys.argv[1]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
        await pg.goto(f"file://{SP}/og/card.html", wait_until="networkidle"); await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{SP}/og/og.png")
        await b.close()
asyncio.run(main())
