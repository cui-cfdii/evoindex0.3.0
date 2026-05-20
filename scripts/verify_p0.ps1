# EvoIndex 3.0 P0 验证脚本 (Windows PowerShell)
# 用法: 在 PowerShell 中运行此脚本

$projectRoot = "\\wsl.localhost\Ubuntu-24.04\home\cuihao\workspace\evoindex0_3_0"

Write-Host "EvoIndex 3.0 P0 验证" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan

# 1. 检查 LM Studio
Write-Host "`n[1/3] 检查 LM Studio..." -ForegroundColor Yellow
try {
    $models = Invoke-RestMethod -Uri "http://127.0.0.1:1234/v1/models" -Method Get -TimeoutSec 5
    $embedModel = $models.data | Where-Object { $_.id -like "*nomic*" -or $_.id -like "*embed*" }
    if ($embedModel) {
        Write-Host "  ✅ LM Studio 运行中，嵌入模型: $($embedModel.id)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ LM Studio 运行中，但未找到嵌入模型。运行: lms load text-embedding-nomic-embed-text-v1.5" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ❌ LM Studio 不可用: $_" -ForegroundColor Red
    Write-Host "  运行: lms server start" -ForegroundColor Yellow
    exit 1
}

# 2. 测试嵌入
Write-Host "`n[2/3] 测试嵌入 API..." -ForegroundColor Yellow
try {
    $body = @{ model = "text-embedding-nomic-embed-text-v1.5"; input = "测试嵌入" } | ConvertTo-Json
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:1234/v1/embeddings" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 10
    $dim = $resp.data[0].embedding.Count
    Write-Host "  ✅ 嵌入成功！向量维度: $dim" -ForegroundColor Green
} catch {
    Write-Host "  ❌ 嵌入失败: $_" -ForegroundColor Red
    exit 1
}

# 3. 运行 P0 测试
Write-Host "`n[3/3] 运行 P0 测试..." -ForegroundColor Yellow
Write-Host "  (在 Node.js 中运行，需要先 cd 到项目目录)"
Write-Host ""
Write-Host "  在 PowerShell 中执行:" -ForegroundColor Cyan
Write-Host "  cd \\\\wsl.localhost\\Ubuntu-24.04\\home\\cuihao\\workspace\\evoindex0_3_0"
Write-Host "  node test\p0_test.mjs \\\\wsl.localhost\\Ubuntu-24.04\\home\\cuihao\\workspace\\evoindex0_3_0\\..\\..\\tmp\\quick_index.json"
Write-Host ""
Write-Host "  或使用 EvoIndex 2.0 测试索引:"
Write-Host "  node test\p0_test.mjs C:\Users\cuihao\EvoIndex-2-0\test_index.json"
Write-Host ""

Write-Host "✅ 验证脚本完成" -ForegroundColor Green
