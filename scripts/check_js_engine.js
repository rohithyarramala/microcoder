const fs=require('fs'), path=require('path');
const [engine, dir, n0] = process.argv.slice(2);
eval(fs.readFileSync(engine,'utf8')+'\nglobalThis.MC=MC;');
const ab=b=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);
const mc=MC(ab(fs.readFileSync(path.join(dir,'model.bin'))), ab(fs.readFileSync(path.join(dir,'tok.bin'))));
const P=JSON.parse(fs.readFileSync(path.join(dir,'prompts.json'),'utf8'));
const n=Math.min(Number(n0)||P.length,P.length);
function sig(s){const m=/([A-Za-z_]\w*)\(([^()]*)\)/.exec(s);return m?`def ${m[1]}(${m[2]}):`:"";}
let out=[];
for(let i=0;i<n;i++){const pre=sig(P[i]); const g=mc.generate("<q>"+P[i]+"<c>"+pre,120,0); out.push(g===null?null:pre+g);}
fs.writeFileSync(path.join(dir,'js_out.json'),JSON.stringify(out));
console.log('generated',n);
