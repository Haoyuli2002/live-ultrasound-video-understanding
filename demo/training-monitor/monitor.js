const jobs = [["bs4",5963020,"#68dfbd"],["bs8",5963195,"#ffc377"]];
let data = [];
const get = async file => {
  const res = await fetch(file,{cache:"no-store"});
  if (!res.ok) throw new Error(file);
  return res.text();
};
async function load([key,id,color]) {
  const [log,csv] = await Promise.all([get(key+"_training.log"),get(key+"_gpu.csv")]);
  const entries = log.split("\n").flatMap(line=>{try{return [JSON.parse(line)]}catch{return []}});
  const epochs = entries.filter(r=>Number.isFinite(r.train_loss));
  const samples = csv.trim().split("\n").slice(1).flatMap(line=>{
    const f=line.split(",").map(x=>x.trim());
    if(f.length!==5 || f.some(x=>!x)) return [];
    const time=Date.parse(f[0].replaceAll("/","-").replace(" ","T")+"Z");
    const row=[time,Number(f[4]),Number(f[2])/1024,Number(f[3])/1024];
    return row.every(Number.isFinite)?[row]:[];
  });
  const bins=new Map();
  for(const [time,gpu,mem] of samples){
    const n=Math.floor((time-samples[0][0])/30000);
    const b=bins.get(n)||{x:n/2,sum:0,count:0,peak:0};
    b.sum+=gpu;b.count++;b.peak=Math.max(b.peak,mem);bins.set(n,b);
  }
  return {key,id,color,epochs,samples,bins:[...bins.values()],done:entries.some(r=>r.first_answer)};
}
function chart(id,series,{xmax,ymin=0,ymax,xlabel,ylabel,log=false,epoch=false}){
  const el=document.getElementById(id), w=el.clientWidth,h=el.clientHeight,dpr=window.devicePixelRatio||1;
  el.width=Math.round(w*dpr);el.height=Math.round(h*dpr);
  const c=el.getContext("2d");c.scale(dpr,dpr);
  const l=65,r=w-22,t=15,b=h-45;
  const coordX=x=>l+(r-l)*x/xmax;
  const unit=y=>log?Math.log10(Math.max(y,ymin)):y;
  const coordY=y=>b-(b-t)*(unit(y)-unit(ymin))/(unit(ymax)-unit(ymin));
  c.font="12px system-ui";c.fillStyle="#b4c5c9";c.textAlign="right";
  const ticks=log?Array.from({length:1+Math.round(unit(ymax)-unit(ymin))},(_,i)=>10**(unit(ymin)+i)):
    Array.from({length:5},(_,i)=>ymin+(ymax-ymin)*i/4);
  for(const value of ticks){
    const y=coordY(value);c.strokeStyle="#304047";c.beginPath();c.moveTo(l,y);c.lineTo(r,y);c.stroke();
    c.fillText(Number(value.toPrecision(3)).toString(),l-8,y+4);
  }
  c.strokeStyle="#768b92";c.beginPath();c.moveTo(l,t);c.lineTo(l,b);c.lineTo(r,b);c.stroke();
  c.textAlign="center";
  const xs=epoch?[0,5,10,15,20]:Array.from({length:5},(_,i)=>xmax*i/4);
  for(const x of xs){const px=coordX(x);c.fillText(String(+x.toFixed(1)),px,b+16)}
  c.fillText(xlabel,(r+l)/2,h-4);
  c.save();c.translate(13,(b+t)/2);c.rotate(-Math.PI/2);c.fillText(ylabel,0,0);c.restore();
  for(const [color,points] of series){
    c.strokeStyle=color;c.fillStyle=color;c.lineWidth=2;c.beginPath();
    points.forEach(([x,y],i)=>{if(i)c.lineTo(coordX(x),coordY(y));else c.moveTo(coordX(x),coordY(y))});c.stroke();
    if(epoch)for(const [x,y] of points){c.beginPath();c.arc(coordX(x),coordY(y),3,0,2*Math.PI);c.fill()}
  }
}
function draw(){
  if(!data.length)return;
  const series=fn=>data.map(d=>[d.color,fn(d)]);
  const losses=data.flatMap(d=>d.epochs.map(e=>e.train_loss));
  const log=document.getElementById("scale").value==="log";
  const ymin=log?10**Math.floor(Math.log10(Math.min(0.1,...losses.filter(x=>x>0)))):0;
  const ymax=log?10**Math.ceil(Math.log10(Math.max(1,...losses))):Math.ceil(Math.max(1,...losses));
  chart("loss",series(d=>d.epochs.map(e=>[e.epoch,e.train_loss])),{xmax:20,ymin,ymax,xlabel:"训练轮数（epoch）",ylabel:"NTP loss",epoch:true,log});
  const xmax=4*Math.ceil(Math.max(4,...data.map(d=>d.bins.at(-1)?.x||0))/4);
  chart("gpu",series(d=>d.bins.map(b=>[b.x,b.sum/b.count])),{xmax,ymax:100,xlabel:"作业运行时间（分钟）",ylabel:"利用率（%）"});
  chart("mem",series(d=>d.bins.map(b=>[b.x,b.peak])),{xmax,ymax:Math.ceil(Math.max(1,...data.map(d=>d.samples.at(-1)?.[3]||0))),xlabel:"作业运行时间（分钟）",ylabel:"显存（GiB）"});
}
async function refresh(){
  try{
    const [runs,status]=await Promise.all([Promise.all(jobs.map(load)),get("sync.txt")]);data=runs;
    document.querySelector("#runs").innerHTML=data.map(d=>{
      const e=d.epochs.at(-1), g=d.samples.at(-1), n=e?.epoch||0;
      return '<article class="run" style="--accent:'+d.color+'"><div class="run-head"><h2>'+d.key.toUpperCase()+'</h2><small>Slurm '+d.id+'</small></div><div class="metrics"><div><small>已完成轮数</small><b>'+n+'/20</b></div><div><small>NTP loss</small><b>'+(e?e.train_loss.toFixed(4):'—')+'</b></div><div><small>最新显存 · GiB</small><b>'+(g?g[2].toFixed(1):'—')+'</b></div></div><div class="bar"><span style="width:'+n*5+'%"></span></div><small>'+(d.done?'回放已完成':'已记录 '+(e?.optimizer_steps||0)+' 步')+'　<a href="'+d.key+'_training.log">日志</a><a href="'+d.key+'_gpu.csv">GPU CSV</a></small></article>';
    }).join("");
    document.querySelector("#status").textContent=status;
    document.querySelector("#refresh").textContent="10 秒刷新 · "+new Date().toLocaleTimeString();
    draw();
  }catch(err){document.querySelector("#status").textContent="读取失败 · "+err.message}
  setTimeout(refresh,10000);
}
document.getElementById("scale").addEventListener("change",draw);
window.addEventListener("resize",draw);
refresh();
