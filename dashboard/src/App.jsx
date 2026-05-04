import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ShieldAlert, ShieldCheck, Activity, BarChart3, 
  Clock, CreditCard, ChevronRight, Search, Zap, TrendingDown, Layers, Info, GitCompare
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell, AreaChart, Area, Legend, LineChart, Line
} from 'recharts';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || "/api";

const formatNum = (val, decimals = 1) => {
  if (val === null || val === undefined || isNaN(val)) return "0.0";
  return val.toFixed(decimals);
};

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('Monitor');
  const [modelType, setModelType] = useState('sparse'); // Default to best model
  const [showComparison, setShowComparison] = useState(false);
  const [threshold, setThreshold] = useState(0.035);
  const [transactions, setTransactions] = useState([]);
  const [selectedTx, setSelectedTx] = useState(null);
  const [shapData, setShapData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reviewedTxs, setReviewedTxs] = useState(new Set());
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    fetchTransactions();
    fetchMetrics();
    fetchModelInfo();
    const interval = setInterval(fetchTransactions, 30000);
    return () => clearInterval(interval);
  }, [modelType]);

  const fetchModelInfo = async () => {
    try {
      const res = await axios.get(`${API_BASE}/model-info`);
      setModelInfo(res.data);
    } catch (err) {
      console.error("Failed to fetch model info");
    }
  };

  const fetchTransactions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/transactions?limit=12`);
      const tdata = res.data;
      const ids = tdata.map(t => t.id);
      const predRes = await axios.post(`${API_BASE}/predict?model_type=${modelType}`, ids);
      
      const combined = tdata.map(t => ({
        ...t,
        ...(predRes.data.find(p => p.id === t.id) || {})
      }));
      setTransactions(combined);
    } catch (err) {
      console.error("Failed to fetch transactions");
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await axios.get(`${API_BASE}/metrics`);
      setMetrics(res.data);
    } catch (err) {
      console.error("Failed to fetch metrics");
    }
  };

  const handleSelect = async (tx) => {
    setSelectedTx(tx);
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/explain?tid=${tx.id}&model_type=${modelType}`);
      setShapData(res.data);
    } catch (err) {
      setShapData([]);
    } finally {
      setLoading(false);
    }
  };

  const renderContent = () => {
    if (showComparison) {
      return <ComparisonView metrics={metrics} modelInfo={modelInfo} onClose={() => setShowComparison(false)} />;
    }

    return (
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab + modelType}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4 }}
          className="w-full h-full"
        >
          {activeTab === 'Monitor' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {transactions.map((tx) => {
                const isDynamicFraud = tx.score !== undefined ? tx.score > threshold : tx.is_fraud;
                return (
                  <TransactionCard 
                    key={tx.id} 
                    tx={{ ...tx, is_fraud: isDynamicFraud }} 
                    onClick={() => handleSelect({ ...tx, is_fraud: isDynamicFraud })}
                    selected={selectedTx?.id === tx.id}
                    isReviewed={reviewedTxs.has(tx.id)}
                  />
                );
              })}
            </div>
          )}
          {activeTab === 'Performance' && <PerformanceView metrics={metrics} currentModel={modelType} />}
          {activeTab === 'Alert Rules' && <AlertRulesView threshold={threshold} setThreshold={setThreshold} modelType={modelType} />}
        </motion.div>
      </AnimatePresence>
    );
  };

  return (
    <div className="flex h-screen w-full bg-[#f8fafc] overflow-hidden text-slate-800 font-sans">
      <aside className="w-72 flex flex-col glass-sidebar shadow-2xl z-30">
        <div className="p-8 flex items-center space-x-3 border-b border-white/20">
          <motion.div 
            whileHover={{ rotate: 180 }}
            className="p-2 bg-gradient-to-br from-indigo-600 to-indigo-800 rounded-xl shadow-lg shadow-indigo-200"
          >
            <ShieldAlert className="text-white w-6 h-6" />
          </motion.div>
          <div>
            <h1 className="text-xl font-black tracking-tight text-slate-900 leading-none flex items-center">
              FINTRAC<span className="text-orange-600 ml-0.5">-AI</span>
            </h1>
            <p className="text-[9px] font-bold text-slate-500 uppercase tracking-[0.2em] mt-1">Anomaly Detection</p>
          </div>
        </div>
        
        <nav className="flex-1 py-8 px-4 space-y-2">
          <NavItem icon={<Activity size={20}/>} label="Threat Monitor" active={activeTab === 'Monitor' && !showComparison} onClick={() => {setActiveTab('Monitor'); setShowComparison(false);}} />
          <NavItem icon={<BarChart3 size={20}/>} label="Model Insights" active={activeTab === 'Performance' && !showComparison} onClick={() => {setActiveTab('Performance'); setShowComparison(false);}} />
          <NavItem icon={<GitCompare size={20}/>} label="Cross-Model Bench" active={showComparison} onClick={() => setShowComparison(true)} />
          <NavItem icon={<ShieldAlert size={20}/>} label="Security Rules" active={activeTab === 'Alert Rules' && !showComparison} onClick={() => {setActiveTab('Alert Rules'); setShowComparison(false);}} />
        </nav>

        <div className="p-8">
          <div className="glass-card p-5 rounded-2xl border border-white/40 space-y-4 bg-white/50">
             <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-[0_0_10px_#22c55e]"></div>
                <span className="text-[10px] font-bold text-slate-600 uppercase">Live Detection active</span>
             </div>
             <div className="flex justify-between items-end">
                <div>
                   <p className="text-[10px] text-slate-400 uppercase font-bold tracking-tighter">Lat. Index</p>
                   <p className="text-lg font-black text-slate-800 font-mono">
                     {metrics?.[modelType]?.latency_ms ? formatNum(metrics[modelType].latency_ms, 1) : '--'}<span className="text-sm font-normal ml-0.5">ms</span>
                   </p>
                </div>
                <div className="text-right">
                   <p className="text-[10px] text-slate-400 uppercase font-bold tracking-tighter">Active Engine</p>
                   <p className="text-[10px] font-black text-indigo-600 uppercase">{modelType}</p>
                </div>
             </div>
          </div>
        </div>
      </aside>

      <main className="flex-1 flex flex-col overflow-hidden relative">
        <header className="h-24 flex items-center justify-between px-12 glass-header z-10 bg-white/80 border-b border-slate-100">
          <div className="flex items-center gap-8">
            <h2 className="text-2xl font-black text-slate-900 tracking-tight">{showComparison ? 'Multi-Model Comparative Analysis' : activeTab}</h2>
            {!showComparison && (
              <div className="flex bg-slate-100 p-1 rounded-2xl shadow-inner">
                 {['standard', 'sparse', 'denoising'].map((m) => (
                   <button 
                      key={m}
                      onClick={() => setModelType(m)}
                      className={`px-6 py-2 rounded-xl text-[10px] font-black uppercase tracking-wider transition-all duration-300 ${modelType === m ? 'bg-white text-indigo-600 shadow-sm scale-105' : 'text-slate-500 hover:text-slate-700'}`}
                   >
                     {m}
                   </button>
                 ))}
              </div>
            )}
          </div>
          <div className="flex items-center space-x-6">
             <div className="text-right">
               <p className="text-[10px] font-bold text-slate-400 uppercase leading-none mb-1">Imbalance Ratio</p>
               <p className="text-xs font-black text-indigo-600">0.172% (Critical)</p>
             </div>
             <div className="w-12 h-12 rounded-2xl bg-slate-900 flex items-center justify-center text-white font-black text-sm shadow-xl border-2 border-white">
                S
             </div>
          </div>
        </header>

        <div className="flex-1 p-12 overflow-y-auto custom-scroll bg-transparent">
          {renderContent()}
        </div>
      </main>

      <AnimatePresence>
        {selectedTx && !showComparison && (
          <motion.aside 
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="w-[520px] glass-sidebar border-l border-slate-200 p-10 flex flex-col shadow-2xl z-40 h-full overflow-hidden bg-white"
          >
            <div className="flex justify-between items-center mb-10 pb-6 border-b border-slate-100">
              <h2 className="text-xl font-black text-slate-800 uppercase tracking-tighter">Forensic Investigation</h2>
              <button onClick={() => setSelectedTx(null)} className="p-3 hover:bg-slate-50 rounded-2xl text-slate-400 transition-all">
                <ChevronRight className="rotate-180" />
              </button>
            </div>

            <div className="space-y-10 flex-1 overflow-y-auto pr-2 custom-scroll pb-10">
              <div className="p-8 rounded-[40px] border border-slate-100 shadow-sm bg-slate-50/50">
                <p className="text-[11px] font-bold text-indigo-600 uppercase mb-6 tracking-widest flex items-center gap-2">
                   <div className="w-1.5 h-1.5 bg-indigo-600 rounded-full animate-ping"></div> Active Engine: {modelType}
                </p>
                <div className="grid grid-cols-2 gap-6">
                  <div className="p-6 bg-white rounded-3xl border border-slate-100 shadow-sm">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Tx Amount</p>
                    <p className="text-2xl font-black text-slate-900">${formatNum(selectedTx.Amount, 2)}</p>
                  </div>
                  <div className={`p-6 rounded-3xl border ${selectedTx.is_fraud ? 'bg-rose-50 border-rose-100' : 'bg-emerald-50 border-emerald-100'}`}>
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Threat Score</p>
                    <p className={`text-2xl font-black ${selectedTx.is_fraud ? 'text-rose-600' : 'text-emerald-600'}`}>
                      {formatNum((selectedTx.score || 0) * 100, 1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-black text-slate-800 mb-8 flex items-center gap-3">
                  <Activity size={20} className="text-indigo-600" /> Explainable AI (XAI) Matrix
                </h3>
                {loading ? (
                  <div className="h-64 flex flex-col items-center justify-center space-y-4">
                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600"></div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Calculating Shapley Values...</p>
                  </div>
                ) : shapData.length > 0 ? (
                  <div className="h-80 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={shapData} layout="vertical" margin={{ left: -10 }}>
                        <XAxis type="number" hide />
                        <YAxis dataKey="feature" type="category" width={50} fontSize={10} tick={{fill: '#94a3b8', fontWeight: 'bold'}} axisLine={false} tickLine={false} />
                        <Tooltip cursor={{fill: 'rgba(0,0,0,0.02)'}} contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 25px rgba(0,0,0,0.1)' }} />
                        <Bar dataKey="value" radius={[0, 12, 12, 0]} barSize={16}>
                          {shapData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#f43f5e' : '#10b981'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-64 flex flex-col items-center justify-center space-y-2 border-2 border-dashed border-slate-100 rounded-[40px]">
                    <Activity className="text-slate-200" size={32} />
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">No XAI Data available</p>
                  </div>
                )}
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
};

const ComparisonView = ({ metrics, modelInfo, onClose }) => {
  if (!metrics) return <div className="h-full flex items-center justify-center text-slate-400 font-bold animate-pulse text-xs uppercase tracking-[0.2em]">SYNCHRONIZING REPOSITORY DATA...</div>;
  
  const mTypes = ['standard', 'sparse', 'denoising'];
  const labels = {
    standard: { tag: 'Baseline', def: 'Maps high-dimensional transactions into a compact bottleneck to learn simple representations.', pros: ['Highest Throughput', 'Stable Reference'] },
    sparse: { tag: 'High-Fidelity', def: 'Forces selective neuron activation to capture precise fraud signatures in imbalanced data.', pros: ['Best AUPRC', 'Noise Filtering'] },
    denoising: { tag: 'Robust', def: 'Trained to recover clean signals from noisy inputs, preventing overfitting on outliers.', pros: ['Generalization', 'Robustness'] }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-12 pb-24"
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {mTypes.map(m => (
          <div key={m} className={`p-8 rounded-[50px] border transition-all duration-500 ${m === 'sparse' ? 'bg-gradient-to-br from-orange-500 to-red-600 text-white shadow-[0_20px_50px_rgba(249,115,22,0.3)] scale-105 z-10' : 'bg-white border-slate-100 shadow-xl hover:shadow-2xl hover:-translate-y-1'}`}>
            <div className="flex justify-between items-start mb-10">
               <div>
                  <h3 className="text-xl font-black tracking-tight mb-2 uppercase flex items-center gap-2">
                    {m === 'sparse' && <Zap size={18} className="text-yellow-300 fill-yellow-300 animate-pulse" />}
                    {m}
                  </h3>
                  <span className={`px-4 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${m === 'sparse' ? 'bg-white/20 text-white backdrop-blur-md' : 'bg-orange-50 text-orange-600'}`}>{labels[m].tag}</span>
               </div>
               <div className="text-right">
                  <p className="text-[10px] font-bold text-slate-500 uppercase mb-1">AUPRC</p>
                  <p className="text-3xl font-black tracking-tighter">{formatNum(metrics[m].auprc * 100, 1)}%</p>
               </div>
            </div>
            
            <p className={`text-xs leading-relaxed mb-10 font-medium ${m === 'sparse' ? 'text-slate-400' : 'text-slate-500'}`}>
              {labels[m].def}
            </p>

            <div className="space-y-4 mb-10">
               <div className="flex justify-between text-[11px] font-bold uppercase tracking-tight border-b border-slate-100 pb-2">
                  <span className="text-slate-400">F1 Score</span>
                  <span className={m === 'sparse' ? 'text-white' : 'text-slate-900'}>{formatNum(metrics[m].f1 * 100, 1)}%</span>
               </div>
               <div className="flex justify-between text-[11px] font-bold uppercase tracking-tight border-b border-slate-100 pb-2">
                  <span className="text-slate-400">Lat. Index</span>
                  <span className={m === 'sparse' ? 'text-white' : 'text-slate-900'}>{formatNum(metrics[m].latency_ms, 1)}ms</span>
               </div>
            </div>

            <div className="flex flex-wrap gap-2">
               {labels[m].pros.map(p => <span key={p} className={`px-3 py-1 rounded-lg text-[9px] font-black uppercase tracking-widest ${m === 'sparse' ? 'bg-white/10 text-orange-100 border border-white/20' : 'bg-orange-500/10 text-orange-600'}`}>{p}</span>)}
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white p-12 rounded-[60px] border border-slate-100 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-orange-500/5 rounded-full blur-3xl -z-10 translate-x-1/2 -translate-y-1/2"></div>
        <h3 className="text-2xl font-black text-slate-900 mb-12 flex items-center gap-4 uppercase tracking-tighter">
           <div className="p-3 bg-orange-50 rounded-2xl">
             <BarChart3 size={24} className="text-orange-600" />
           </div>
           Enterprise Benchmarking Result
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-separate border-spacing-y-4">
            <thead>
              <tr className="text-[11px] font-black text-slate-400 uppercase tracking-widest">
                <th className="px-8 py-4">Metric</th>
                <th className="px-8 py-4">Standard</th>
                <th className="px-8 py-4">Sparse (Optimal)</th>
                <th className="px-8 py-4">Denoising</th>
                <th className="px-8 py-4">Result</th>
              </tr>
            </thead>
            <tbody className="text-sm font-bold">
              <BenchLine label="AUPRC Accuracy" m1={formatNum(metrics.standard.auprc*100, 1)+'%'} m2={formatNum(metrics.sparse.auprc*100, 1)+'%'} m3={formatNum(metrics.denoising.auprc*100, 1)+'%'} result="Sparse Win" highlight />
              <BenchLine label="Balanced F1" m1={formatNum(metrics.standard.f1*100, 1)+'%'} m2={formatNum(metrics.sparse.f1*100, 1)+'%'} m3={formatNum(metrics.denoising.f1*100, 1)+'%'} result="Sparse Win" highlight />
              <BenchLine label="Latency (ms)" m1={formatNum(metrics.standard.latency_ms, 2)} m2={formatNum(metrics.sparse.latency_ms, 2)} m3={formatNum(metrics.denoising.latency_ms, 2)} result={metrics.sparse.latency_ms <= metrics.standard.latency_ms && metrics.sparse.latency_ms <= metrics.denoising.latency_ms ? "Sparse Fast" : metrics.denoising.latency_ms <= metrics.standard.latency_ms ? "Denoising Fast" : "Standard Fast"} />
              <BenchLine label="Noise Resilience" m1="Low" m2="High" m3="Highest" result="Denoising Win" highlight />
            </tbody>
          </table>
        </div>
      </div>
    </motion.div>
  );
};

const BenchLine = ({ label, m1, m2, m3, result, highlight }) => (
  <tr className="hover:translate-x-2 transition-transform duration-300 group">
    <td className="px-8 py-6 bg-slate-50/50 rounded-l-[30px] border-y border-l border-slate-100 text-slate-500 uppercase text-[10px] tracking-widest font-black group-hover:bg-orange-50/50 transition-colors">{label}</td>
    <td className="px-8 py-6 bg-slate-50/50 border-y border-slate-100 text-slate-900 group-hover:bg-orange-50/50 transition-colors">{m1}</td>
    <td className="px-8 py-6 bg-slate-50/50 border-y border-slate-100 text-orange-600 font-black text-lg group-hover:bg-orange-50/50 transition-colors">{m2}</td>
    <td className="px-8 py-6 bg-slate-50/50 border-y border-slate-100 text-slate-900 group-hover:bg-orange-50/50 transition-colors">{m3}</td>
    <td className="px-8 py-6 bg-slate-50/50 rounded-r-[30px] border-y border-r border-slate-100 group-hover:bg-orange-50/50 transition-colors">
      <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black tracking-widest uppercase shadow-sm ${highlight ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-500'}`}>
         {result}
      </div>
    </td>
  </tr>
);


const CompStat = ({ label, val, color }) => (
  <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
    <p className="text-[9px] font-bold text-slate-500 uppercase mb-1">{label}</p>
    <p className={`text-xl font-black ${color} tracking-tighter`}>{val}</p>
  </div>
);

const TableLine = ({ label, std, spr, result, highlight }) => (
  <tr className="group transition-all hover:translate-x-1">
    <td className="px-6 py-6 bg-white/40 rounded-l-3xl border-y border-l border-white/60 text-slate-500">{label}</td>
    <td className="px-6 py-6 bg-white/40 border-y border-white/60 text-slate-800">{std}</td>
    <td className="px-6 py-6 bg-white/40 border-y border-white/60 text-orange-600">{spr}</td>
    <td className={`px-6 py-6 bg-white/40 rounded-r-3xl border-y border-r border-white/60 ${highlight ? 'text-green-600' : 'text-slate-400'}`}>
       <div className="flex items-center gap-2">
          {highlight && <ShieldCheck size={14} />} {result}
       </div>
    </td>
  </tr>
);

const PerformanceView = ({ metrics, currentModel }) => {
  if (!metrics) return (
    <div className="h-96 flex flex-col items-center justify-center space-y-6">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-600"></div>
      <p className="text-sm font-black text-slate-400 uppercase tracking-widest animate-pulse">Synchronizing Engine Metrics...</p>
    </div>
  );
  
  const currentData = metrics[currentModel];
  if (!currentData) return <div className="p-20 text-center font-bold text-slate-400">Model "{currentModel}" data not found.</div>;
  
  const lossData = (currentData.loss_history || []).map((l, i) => ({ epoch: `E${i+1}`, loss: l }));

  return (
    <motion.div 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      className="space-y-12 pb-24"
    >
      <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
        <MetricCard label="AUPRC Accuracy" value={formatNum(currentData.auprc * 100, 1) + "%"} sub="Research Threshold" icon={<ShieldCheck className="text-green-500" />} />
        <MetricCard label="F1-Score" value={formatNum(currentData.f1 * 100, 1) + "%"} sub="Precision/Recall" icon={<Zap className="text-orange-500" />} />
        <MetricCard label="False Pos. Rate" value={formatNum(currentData.fpr * 100, 2) + "%"} sub="System Purity" icon={<TrendingDown className="text-blue-500" />} />
        <MetricCard 
          label="Inference Speed" 
          value={formatNum(currentData.latency_ms, 2) + "ms"} 
          sub="E2E Pipeline Avg" 
          icon={<Activity className="text-indigo-500" />} 
          breakdown={currentData.latency_breakdown}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 glass-card p-10 rounded-[40px] border border-white/60 shadow-xl overflow-hidden">
           <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-10 flex items-center gap-3">
             <BarChart3 size={20} className="text-orange-500" /> Top Influential Features ({currentModel})
           </h3>
           <div className="h-[250px] w-full">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={currentData.feature_importance} layout="vertical">
                 <XAxis type="number" hide />
                 <YAxis dataKey="feature" type="category" width={40} fontSize={10} tick={{fill: '#64748b', fontWeight: 'bold'}} axisLine={false} tickLine={false} />
                 <Tooltip cursor={{fill: 'rgba(255,255,255,0.1)'}} contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 25px rgba(0,0,0,0.1)' }} />
                 <Bar dataKey="importance" fill="#f97316" radius={[0, 10, 10, 0]} barSize={20} />
               </BarChart>
             </ResponsiveContainer>
           </div>
        </div>

        <div className="glass-card p-10 rounded-[40px] border border-white/60 shadow-xl overflow-hidden">
           <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-10 flex items-center gap-3">
             <CreditCard size={20} className="text-orange-500" /> Amount Impact
           </h3>
           <div className="h-[250px] w-full">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={metrics.global?.amount_dist} margin={{ bottom: 20 }}>
                 <XAxis dataKey="range" stroke="#94a3b8" fontSize={9} axisLine={false} tickLine={false} />
                 <Tooltip contentStyle={{ borderRadius: '16px', border: 'none' }} />
                 <Bar dataKey="fraud" fill="#ef4444" radius={[5, 5, 0, 0]} name="Fraud" stackId="a" />
                 <Bar dataKey="normal" fill="#10b981" radius={[5, 5, 0, 0]} name="Normal" stackId="a" />
               </BarChart>
             </ResponsiveContainer>
           </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        <div className="glass-card p-10 rounded-[40px] border border-white/60 shadow-xl overflow-hidden">
           <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-10 flex items-center gap-3">
             <BarChart3 size={20} className="text-orange-500" /> Reconstruction Distribution
           </h3>
           <div className="h-[300px] w-full">
             <ResponsiveContainer width="100%" height="100%">
                <BarChart data={currentData.error_dist} margin={{ bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="bin" stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} dy={10} />
                  <YAxis stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: '20px', border: 'none', boxShadow: '0 20px 50px rgba(0,0,0,0.1)' }} />
                  <Legend verticalAlign="top" height={36} iconType="circle" />
                  <Bar dataKey="normal" fill="#10b981" radius={[8, 8, 0, 0]} name="Normal behavior" />
                  <Bar dataKey="fraud" fill="#ef4444" radius={[8, 8, 0, 0]} name="Fraud pattern" />
                </BarChart>
             </ResponsiveContainer>
           </div>
        </div>

        <div className="glass-card p-10 rounded-[40px] border border-white/60 shadow-xl overflow-hidden">
           <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-10 flex items-center gap-3">
             <TrendingDown size={20} className="text-orange-500" /> Learning Curve Convergence
           </h3>
           <div className="h-[300px] w-full">
             <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={lossData}>
                  <defs>
                    <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: '20px', border: 'none', boxShadow: '0 20px 50px rgba(0,0,0,0.1)' }} />
                  <Area type="monotone" dataKey="loss" stroke="#f97316" strokeWidth={5} fillOpacity={1} fill="url(#colorLoss)" />
                </AreaChart>
             </ResponsiveContainer>
           </div>
        </div>
      </div>
    </motion.div>
  );
};

const AlertRulesView = ({ threshold, setThreshold, modelType }) => (
  <div className="max-w-3xl mx-auto py-16 animate-in-fade">
    <div className="glass-card p-20 rounded-[60px] border border-white/60 shadow-2xl text-center relative overflow-hidden">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-1 bg-gradient-to-r from-transparent via-orange-500 to-transparent"></div>
      <div className="mx-auto w-24 h-24 bg-orange-100/50 rounded-[30px] flex items-center justify-center mb-10 rotate-3 border-2 border-white">
        <ShieldAlert className="text-orange-600 w-12 h-12" />
      </div>
      <h3 className="text-3xl font-black text-slate-900 mb-4 tracking-tighter">Sensitivity Calibration</h3>
      <p className="text-sm text-slate-500 mb-16 font-medium max-w-md mx-auto">Tuning the reconstruction error boundary for the <span className="text-orange-600 font-bold">{modelType}</span> engine.</p>
      
      <div className="bg-slate-50/50 p-12 rounded-[40px] border border-white/50 mb-10 shadow-inner">
        <div className="flex justify-between items-center mb-10 px-4">
           <span className="text-[11px] font-black text-slate-400 uppercase tracking-widest">Decision Threshold</span>
           <motion.span 
             key={threshold}
             initial={{ scale: 0.8, opacity: 0 }}
             animate={{ scale: 1, opacity: 1 }}
             className="text-6xl font-mono font-black text-slate-900"
           >
             {threshold.toFixed(3)}
           </motion.span>
        </div>
        <input 
          type="range" min="0.001" max="0.2" step="0.005" value={threshold} 
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full h-4 bg-slate-200 rounded-full appearance-none cursor-pointer accent-orange-500 shadow-sm"
        />
        <div className="flex justify-between mt-6 px-2 text-[10px] font-black text-slate-400 uppercase tracking-widest">
           <span>Precision Focused</span>
           <span>Recall Focused</span>
        </div>
      </div>
    </div>
  </div>
);

const NavItem = ({ icon, label, active, onClick }) => (
  <button 
    onClick={onClick}
    className={`w-full flex items-center space-x-4 px-8 py-5 rounded-3xl transition-all duration-500 relative group ${
      active 
      ? 'bg-white text-orange-600 shadow-xl shadow-slate-200/50 scale-[1.02]' 
      : 'text-slate-500 hover:bg-white/40 hover:text-slate-900'
    }`}
  >
    {active && (
      <motion.div 
        layoutId="activeNav"
        className="absolute left-2 w-1.5 h-8 bg-orange-600 rounded-full"
      />
    )}
    <div className={`transition-transform duration-500 ${active ? 'scale-110' : 'group-hover:scale-110'}`}>
      {icon}
    </div>
    <span className="text-sm font-black tracking-tight">{label}</span>
  </button>
);

const TransactionCard = ({ tx, onClick, selected, isReviewed }) => (
  <motion.div
    layout
    whileHover={{ y: -5, scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
    onClick={onClick}
    className={`relative p-0.5 rounded-[30px] cursor-pointer transition-all duration-500 overflow-hidden ${
      selected 
      ? 'bg-gradient-to-br from-orange-500 via-red-500 to-orange-600 shadow-[0_20px_50px_rgba(249,115,22,0.3)]' 
      : 'bg-slate-200 hover:bg-orange-400 shadow-xl'
    }`}
  >
    <div className={`relative h-full p-6 rounded-[28px] flex flex-col justify-between backdrop-blur-xl ${selected ? 'bg-white shadow-inner' : 'bg-gradient-to-br from-white to-orange-50/50'}`}>
      
      {/* Background Tech Details */}
      <div className="absolute top-3 right-5 text-[8px] font-mono text-slate-300 opacity-60 select-none">
        0x{tx.id.toString(16).substring(0, 8).toUpperCase() || "A8F9B2C1"}
      </div>

      {/* Header */}
      <div className="flex justify-between items-start mb-6 z-10">
        <div className="flex items-center gap-3">
          <div className={`p-2.5 rounded-xl border ${tx.is_fraud ? 'bg-rose-50 border-rose-200 text-rose-500 shadow-[0_0_20px_rgba(244,63,94,0.3)]' : 'bg-emerald-50 border-emerald-200 text-emerald-500 shadow-sm'}`}>
            {tx.is_fraud ? <ShieldAlert size={18} className={tx.is_fraud ? "animate-pulse" : ""}/> : <ShieldCheck size={18}/>}
          </div>
          <div>
            <p className="text-[8px] font-black text-slate-400 uppercase tracking-widest">Protocol</p>
            <p className="font-mono text-xs font-black text-slate-700 tracking-tight">TCP/Tx</p>
          </div>
        </div>
        
        {isReviewed && (
          <div className="p-1.5 bg-orange-100 border border-orange-200 text-orange-600 rounded-lg">
            <ShieldCheck size={14} />
          </div>
        )}
      </div>

      {/* Amount Data */}
      <div className="mb-8 z-10">
        <p className="text-[9px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-1">Transfer Vol</p>
        <div className="flex items-baseline gap-1">
          <span className="text-orange-500 font-mono text-sm font-bold">$</span>
          <span className="font-mono text-3xl font-black text-slate-800 tracking-tighter">{formatNum(tx.Amount, 2)}</span>
        </div>
      </div>

      {/* Footer / ID */}
      <div className="flex justify-between items-end border-t border-slate-100 pt-5 z-10 mt-auto">
        <div className="w-2/3 pr-2">
          <p className="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-1.5">Hash Identifier</p>
          <p className="font-mono text-[9px] font-bold text-slate-500 truncate bg-white p-2 rounded-lg border border-slate-100 shadow-inner">#{tx.id}</p>
        </div>
        
        <div className="w-1/3 text-right">
          <span className={`inline-block px-3 py-1.5 rounded-xl text-[8px] font-black uppercase tracking-widest border shadow-sm ${
            tx.is_fraud 
            ? 'bg-gradient-to-r from-rose-500 to-red-500 text-white border-rose-600 shadow-[0_0_15px_rgba(244,63,94,0.4)]' 
            : 'bg-emerald-50 text-emerald-600 border-emerald-200'
          }`}>
            {tx.is_fraud ? 'ANOMALY' : 'SECURE'}
          </span>
        </div>
      </div>
    </div>
  </motion.div>
);

const MetricCard = ({ label, value, sub, icon, breakdown }) => (
  <motion.div 
    whileHover={{ y: -5 }}
    className="glass-card p-8 rounded-[35px] border border-white/60 shadow-xl transition-all group relative overflow-hidden"
  >
    <div className="flex items-center gap-4 mb-6">
       <div className="p-3 bg-slate-50 text-slate-600 rounded-2xl border border-slate-100 group-hover:bg-indigo-600 group-hover:text-white group-hover:shadow-lg group-hover:shadow-indigo-200 transition-all duration-500">
          {React.cloneElement(icon, { size: 20 })}
       </div>
       <div>
         <p className="text-[11px] font-black text-slate-400 uppercase tracking-widest leading-none">{label}</p>
         <p className="text-[10px] font-medium text-slate-400 mt-1.5">{sub}</p>
       </div>
    </div>
    <h4 className="text-3xl font-black text-slate-900 tracking-tighter">{value}</h4>
    
    {breakdown && (
      <div className="mt-6 pt-6 border-t border-slate-50 grid grid-cols-3 gap-2">
         <div className="text-center">
            <p className="text-[8px] font-bold text-slate-400 uppercase">Pre</p>
            <p className="text-[10px] font-black text-indigo-600">{formatNum(breakdown.preprocess_ms, 2)}ms</p>
         </div>
         <div className="text-center">
            <p className="text-[8px] font-bold text-slate-400 uppercase">Inf</p>
            <p className="text-[10px] font-black text-indigo-600">{formatNum(breakdown.inference_ms, 2)}ms</p>
         </div>
         <div className="text-center">
            <p className="text-[8px] font-bold text-slate-400 uppercase">Post</p>
            <p className="text-[10px] font-black text-indigo-600">{formatNum(breakdown.postprocess_ms, 2)}ms</p>
         </div>
      </div>
    )}
  </motion.div>
);

export default Dashboard;
