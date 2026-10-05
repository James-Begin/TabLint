"""Real stdio handshake + calls against the Proofread MCP server (replay tools; live tool must be gated)."""
import asyncio, json, os, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
ROOT = Path(__file__).resolve().parents[1]


async def main():
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PROOFREAD_REPORT_DIR': str(ROOT / 'results' / 'proofread_demos'),
           'PROOFREAD_BENCHMARK': str(ROOT / 'results' / 'proofread' / 'summary.json')}
    env.pop('PROOFREAD_ALLOW_INFERENCE', None)
    params = StdioServerParameters(command=sys.executable, args=['-m', 'proofread.mcp_server'], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print('registered:', [t.name for t in (await s.list_tools()).tools])
            res = await s.call_tool('list_reports', {}); assert not res.isError, res
            reports = json.loads(res.content[0].text) if res.content else []
            rid = (reports[0] if isinstance(reports, list) and reports else json.loads(res.content[0].text))['report_id']
            print('list_reports passed:', rid)
            res = await s.call_tool('get_issues', {'report_id': rid, 'top': 5}); assert not res.isError, res
            print('get_issues passed')
            res = await s.call_tool('report_markdown', {'report_id': rid}); assert not res.isError, res
            print('report_markdown passed')
            res = await s.call_tool('get_issues', {'report_id': '../../etc/passwd'}); assert res.isError
            print('path traversal rejected')
            res = await s.call_tool('proofread_csv', {'csv_path': 'x.csv'}); assert res.isError
            print('live inference gate passed')


if __name__ == '__main__':
    asyncio.run(main())
