import asyncio, sys
from playwright.async_api import async_playwright
SP = sys.argv[1]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = await b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
        await pg.goto("https://lfpformer-nyu.vercel.app/?rotate=0", wait_until="networkidle", timeout=120000)
        await pg.wait_for_timeout(7000)
        await pg.evaluate("""() => { for (const s of ['#viewer-overlay','.viewer-tools','.viewer-legend']) document.querySelector(s).style.display='none';
                                    document.querySelector('#viewer').style.background='transparent'; document.querySelector('#viewer').style.border='0'; }""")
        await pg.wait_for_timeout(1500)
        el = await pg.query_selector("#viewer")
        await el.screenshot(path=f"{SP}/og/brain.png", timeout=120000, omit_background=True)
        await b.close()
asyncio.run(main())
