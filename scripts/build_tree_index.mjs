/**
 * EvoIndex 3.0 — 快速树索引构建器
 * 从 Markdown 文章目录构建可检索的树索引
 *
 * 用法: node scripts/build_tree_index.mjs <articles_dir> <output.json>
 */

import fs from 'fs';
import path from 'path';

const articlesDir = process.argv[2];
const outputPath = process.argv[3];

if (!articlesDir || !outputPath) {
  console.log('用法: node scripts/build_tree_index.mjs <articles_dir> <output.json>');
  process.exit(1);
}

// 收集所有 MD 文件
function collectFiles(dir) {
  const files = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectFiles(fullPath));
    } else if (entry.name.endsWith('.md') || entry.name.endsWith('.txt')) {
      files.push(fullPath);
    }
  }
  return files;
}

// 解析单篇 Markdown
function parseMarkdown(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  let docTitle = path.basename(filePath, path.extname(filePath)).replace(/^[\d_]+/g, '').replace(/_/g, ' ').trim() || 'Untitled';
  const root = { title: docTitle, level: 1, children: [], content: '', _id: 'doc-' + path.basename(filePath) };

  let currentSection = root;
  const levelStack = [{ node: root, level: 1 }];

  let sectionIdx = 0;
  for (const line of lines) {
    const hMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (hMatch) {
      const level = hMatch[1].length + 1; // +1 because doc title is level 1
      const title = hMatch[2].trim();

      // Find parent
      while (levelStack.length > 1 && levelStack[levelStack.length - 1].level >= level) {
        levelStack.pop();
      }
      const parent = levelStack[levelStack.length - 1].node;

      const node = {
        title,
        level,
        children: [],
        content: '',
        _id: `sec-${path.basename(filePath)}-${sectionIdx++}`
      };

      parent.children.push(node);
      levelStack.push({ node, level });
      currentSection = node;
    } else if (line.trim()) {
      currentSection.content += line.trim() + '\n';
    }
  }

  // Clean empty content
  function cleanNode(node) {
    node.content = node.content.trim();
    for (const child of node.children) {
      cleanNode(child);
    }
  }
  cleanNode(root);

  return root;
}

// Main
console.log(`📂 扫描文章目录: ${articlesDir}`);

const files = collectFiles(articlesDir);
console.log(`   找到 ${files.length} 篇文章`);

// 限制数量（pharma 有 1600+，用采样）
let targetFiles = files;
const MAX_FILES = 200;

if (files.length > MAX_FILES) {
  // 均匀采样 + 确保每个子目录都有代表
  const byDir = {};
  for (const f of files) {
    const dir = path.dirname(f);
    if (!byDir[dir]) byDir[dir] = [];
    byDir[dir].push(f);
  }

  targetFiles = [];
  const dirs = Object.keys(byDir);
  const perDir = Math.ceil(MAX_FILES / dirs.length);
  for (const dir of dirs) {
    const dirFiles = byDir[dir];
    const sample = dirFiles.slice(0, Math.min(perDir, dirFiles.length));
    targetFiles.push(...sample);
  }
  console.log(`   采样 ${targetFiles.length} 篇 (原始 ${files.length})`);
}

// 构建树索引
const children = [];
for (const f of targetFiles) {
  try {
    const doc = parseMarkdown(f);
    children.push(doc);
  } catch (err) {
    console.warn(`   ⚠️ 跳过 ${path.basename(f)}: ${err.message}`);
  }
}

const index = {
  version: '3.0',
  root: {
    title: '知识库',
    _id: 'root',
    level: 0,
    children,
    content: '',
  },
  stats: {
    totalDocs: children.length,
    totalNodes: countNodes({ children }),
    builtAt: new Date().toISOString(),
  },
};

function countNodes(node) {
  let count = 1;
  if (node.children) {
    for (const c of node.children) count += countNodes(c);
  }
  return count;
}

fs.writeFileSync(outputPath, JSON.stringify(index, null, 2), 'utf-8');
console.log(`✅ 索引构建完成: ${index.stats.totalNodes} 个节点`);
console.log(`   ${outputPath}`);
