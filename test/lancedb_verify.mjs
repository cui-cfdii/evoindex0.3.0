/**
 * LanceDB Windows 侧验证 — 快速检测原生模块是否正常加载
 * 运行方式 (Windows PowerShell):
 *   node test/lancedb_verify.mjs
 */

async function main() {
  console.log('🔍 LanceDB 原生模块验证\n');
  console.log(`平台: ${process.platform} ${process.arch}`);
  console.log(`Node.js: ${process.version}\n`);

  // 1. 检查模块文件是否存在
  const fs = await import('fs');
  const path = await import('path');

  const modulePath = path.join(
    import.meta.dirname, '..', 'node_modules',
    '@lancedb', 'vectordb-win32-x64-msvc', 'index.node'
  );
  console.log(`原生模块路径: ${modulePath}`);
  console.log(`文件存在: ${fs.existsSync(modulePath) ? '✅' : '❌'}`);
  if (fs.existsSync(modulePath)) {
    const stat = fs.statSync(modulePath);
    console.log(`文件大小: ${(stat.size / 1024 / 1024).toFixed(1)} MB\n`);
  }

  // 2. 尝试加载 vectordb
  console.log('─'.repeat(50));
  console.log('加载 vectordb...');
  try {
    const lancedb = await import('vectordb');
    console.log('✅ vectordb 导入成功');

    // 3. 尝试创建内存数据库
    const db = await lancedb.connect('lance_verify_test');
    console.log('✅ 数据库连接成功');

    // 4. 尝试创建表 + 写入 + 搜索
    const testData = [
      { id: 1, vec: Array(768).fill(0.1), label: 'test_a' },
      { id: 2, vec: Array(768).fill(0.5), label: 'test_b' },
      { id: 3, vec: Array(768).fill(0.9), label: 'test_c' },
    ];
    const table = await db.createTable('verify', testData);
    console.log(`✅ 表创建成功 (${testData.length} 条)`);

    const results = await table
      .search(Array(768).fill(0.5))
      .limit(2)
      .execute();
    console.log(`✅ 向量搜索成功 (返回 ${results.length} 条)`);
    console.log(`   Top-1: id=${results[0].id}, label=${results[0].label}, dist=${results[0]._distance?.toFixed(4)}`);

    // 清理
    await db.dropTable('verify');

    console.log('\n' + '═'.repeat(50));
    console.log('🎉 LanceDB Windows 原生模块工作正常！');
  } catch (err) {
    console.log(`❌ 失败: ${err.message}`);
    console.log(`\n错误详情: ${err.stack?.split('\n').slice(0, 3).join('\n')}`);
    process.exit(1);
  }
}

main();
