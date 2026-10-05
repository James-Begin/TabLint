"""Actual stdio handshake and read-only calls against the local evidence server."""
import asyncio
import os
from pathlib import Path
import sys
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
ROOT=Path(__file__).resolve().parents[1]


async def main():
    env={**os.environ,'PYTHONPATH':str(ROOT),'AUDIT_REPORT_DIR':str(ROOT/'results'/'reports'),
         'AUDIT_BENCHMARK_DIR':str(ROOT/'results'/'confirmation')}
    params=StdioServerParameters(command=sys.executable,args=['-m','auditkit.mcp_server'],env=env)
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            tools=await session.list_tools()
            print('registered:',[t.name for t in tools.tools])
            result=await session.call_tool('list_audits',{})
            assert not result.isError,result
            print('list_audits stdio call passed')
            result=await session.call_tool('inspect_audit',{'report_id':'breast_101_0.json'})
            assert not result.isError,result
            print('inspect_audit real artifact call passed')
            result=await session.call_tool('benchmark_summary',{})
            assert not result.isError,result
            print('benchmark_summary call passed')
            result=await session.call_tool('generate_demo_audit',{})
            assert result.isError,'live inference should be disabled by default'
            print('live inference opt-in guard passed')


if __name__=='__main__':asyncio.run(main())
