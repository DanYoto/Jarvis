import asyncio
from server import simple_query, complex_query, auto_route_query, analyze_query_complexity

async def interactive_test():
    """交互式测试工具"""
    
    while True:
        print("\n" + "="*50)
        print("MCP Server 交互式测试")
        print("="*50)
        print("1. 测试 Simple Query")
        print("2. 测试 Complex Query")
        print("3. 测试 Auto Route Query")
        print("4. 测试 Complexity Analysis")
        print("5. 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == '5':
            print("退出测试")
            break
        
        if choice not in ['1', '2', '3', '4']:
            print("无效选择，请重试")
            continue
        
        query = input("请输入查询内容: ").strip()
        
        if not query:
            print("查询不能为空")
            continue
        
        print("\n处理中...")
        
        try:
            if choice == '1':
                result = await simple_query(query)
            elif choice == '2':
                result = await complex_query(query)
            elif choice == '3':
                result = await auto_route_query(query)
            elif choice == '4':
                result = await analyze_query_complexity(query)
            
            print("\n结果:")
            print("-" * 50)
            print(result)
            print("-" * 50)
            
        except Exception as e:
            print(f"\n错误: {e}")

if __name__ == "__main__":
    asyncio.run(interactive_test())