/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { 
  Users, 
  AlertTriangle, 
  TrendingUp, 
  Building2, 
  Calendar, 
  Search, 
  Filter, 
  Download, 
  RefreshCw,
  ChevronRight,
  MapPin,
  Briefcase,
  Clock
} from 'lucide-react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  BarChart, Bar
} from 'recharts';
import { motion } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- UTILS ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- COMPONENTS ---

const StatCard = ({ title, value, trend, icon: Icon, color }: any) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex items-start justify-between"
  >
    <div>
      <p className="text-sm font-medium text-slate-500 mb-1">{title}</p>
      <h3 className="text-2xl font-bold text-slate-900">{value}</h3>
      {trend && (
        <p className={cn("text-xs mt-2 flex items-center gap-1", trend.startsWith('+') ? "text-red-500" : "text-emerald-500")}>
          <TrendingUp className="w-3 h-3" />
          {trend} so với tháng trước
        </p>
      )}
    </div>
    <div className={cn("p-3 rounded-lg", color)}>
      <Icon className="w-6 h-6 text-white" />
    </div>
  </motion.div>
);

export default function App() {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [mlData, setMlData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const fetchPredictions = async () => {
    try {
      const response = await fetch('/api/ml/predictions');
      const data = await response.json();
      setMlData(data);
    } catch (error) {
      console.error("Error fetching predictions:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, []);

  const handleRunPipeline = async () => {
    setIsRefreshing(true);
    try {
      const response = await fetch('/api/ml/run', { method: 'POST' });
      const result = await response.json();
      console.log("Pipeline result:", result);
      await fetchPredictions();
    } catch (error) {
      console.error("Error running pipeline:", error);
    } finally {
      setIsRefreshing(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <RefreshCw className="w-12 h-12 text-indigo-600 animate-spin" />
          <p className="text-slate-500 font-medium">Đang tải dữ liệu dự báo...</p>
        </div>
      </div>
    );
  }

  const stats = mlData || {
    total_employees: 0,
    high_risk_count: 0,
    avg_risk_score: 0,
    risk_distribution: [],
    top_high_risk: []
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold leading-tight">Attrition Predictor</h1>
              <p className="text-xs text-slate-500">Hệ thống dự báo rủi ro nghỉ việc sớm</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-2 text-sm text-slate-500 bg-slate-100 px-3 py-1.5 rounded-full">
              <Calendar className="w-4 h-4" />
              <span>Dữ liệu cập nhật: 01/03/2025</span>
            </div>
            <button 
              onClick={handleRunPipeline}
              disabled={isRefreshing}
              className={cn(
                "flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all",
                isRefreshing && "opacity-70 cursor-not-allowed"
              )}
            >
              <RefreshCw className={cn("w-4 h-4", isRefreshing && "animate-spin")} />
              {isRefreshing ? "Đang chạy MLOps Pipeline..." : "Chạy MLOps Pipeline"}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard 
            title="Tổng nhân sự Active" 
            value={stats.total_employees} 
            icon={Users} 
            color="bg-blue-500" 
          />
          <StatCard 
            title="Nhân sự Rủi ro cao" 
            value={stats.high_risk_count} 
            trend="+2.4%"
            icon={AlertTriangle} 
            color="bg-red-500" 
          />
          <StatCard 
            title="Điểm rủi ro TB" 
            value={`${stats.avg_risk_score}/100`} 
            icon={TrendingUp} 
            color="bg-amber-500" 
          />
          <StatCard 
            title="Doanh nghiệp đối tác" 
            value={4} 
            icon={Building2} 
            color="bg-indigo-500" 
          />
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Risk Distribution */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm lg:col-span-1">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-6">Phân bổ rủi ro</h3>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={stats.risk_distribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {stats.risk_distribution.map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend verticalAlign="bottom" height={36}/>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Attrition Trend */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm lg:col-span-2">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-6">Xu hướng rủi ro nghỉ việc</h3>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={[
                  { month: 'Oct 24', risk: 38 },
                  { month: 'Nov 24', risk: 40 },
                  { month: 'Dec 24', risk: 45 },
                  { month: 'Jan 25', risk: 42 },
                  { month: 'Feb 25', risk: 44 },
                  { month: 'Mar 25', risk: 46 },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="month" axisLine={false} tickLine={false} tick={{fontSize: 12, fill: '#94a3b8'}} />
                  <YAxis axisLine={false} tickLine={false} tick={{fontSize: 12, fill: '#94a3b8'}} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="risk" 
                    stroke="#4f46e5" 
                    strokeWidth={3} 
                    dot={{ r: 4, fill: '#4f46e5', strokeWidth: 2, stroke: '#fff' }}
                    activeDot={{ r: 6, strokeWidth: 0 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Enterprise & High Risk List */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Enterprise Risk */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm lg:col-span-1">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-6">Rủi ro theo Doanh nghiệp</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={[
                  { company: 'TechCorp', risk: 12 },
                  { company: 'RetailCo', risk: 25 },
                  { company: 'FactoryX', risk: 18 },
                  { company: 'SalesForce', risk: 32 },
                ]} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" hide />
                  <YAxis 
                    dataKey="company" 
                    type="category" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{fontSize: 12, fill: '#475569'}}
                    width={80}
                  />
                  <Tooltip cursor={{fill: 'transparent'}} />
                  <Bar dataKey="risk" fill="#6366f1" radius={[0, 4, 4, 0]} barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 pt-4 border-t border-slate-100">
              <p className="text-xs text-slate-500 italic">
                * Tỷ lệ nhân sự rủi ro cao trên tổng quy mô doanh nghiệp.
              </p>
            </div>
          </div>

          {/* High Risk Table */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm lg:col-span-2 overflow-hidden">
            <div className="p-6 border-b border-slate-100 flex items-center justify-between">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">Danh sách rủi ro cao (Top 5)</h3>
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input 
                    type="text" 
                    placeholder="Tìm nhân viên..." 
                    className="pl-9 pr-4 py-1.5 bg-slate-100 border-none rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 w-48"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                </div>
                <button className="p-2 hover:bg-slate-100 rounded-lg text-slate-500">
                  <Filter className="w-4 h-4" />
                </button>
                <button className="p-2 hover:bg-slate-100 rounded-lg text-slate-500">
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50 text-[11px] uppercase tracking-wider text-slate-500 font-bold">
                    <th className="px-6 py-4">Nhân viên</th>
                    <th className="px-6 py-4">Doanh nghiệp</th>
                    <th className="px-6 py-4">Phòng ban</th>
                    <th className="px-6 py-4">Thâm niên</th>
                    <th className="px-6 py-4">Điểm rủi ro</th>
                    <th className="px-6 py-4 text-right">Hành động</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {stats.top_high_risk.map((emp: any) => (
                    <tr key={emp.emp_id} className="hover:bg-slate-50 transition-colors group">
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-slate-600 font-bold text-xs">
                            {emp.emp_id.split('_').pop()?.charAt(0)}
                          </div>
                          <div>
                            <div className="text-sm font-bold text-slate-900">{emp.emp_id}</div>
                            <div className="text-[10px] text-slate-400 font-mono">{emp.emp_id}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">{emp.company}</td>
                      <td className="px-6 py-4 text-sm text-slate-600">{emp.dept}</td>
                      <td className="px-6 py-4 text-sm text-slate-600">{emp.tenure_months} months</td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 w-16 bg-slate-100 rounded-full overflow-hidden">
                            <div 
                              className={cn(
                                "h-full rounded-full",
                                emp.risk_score > 80 ? "bg-red-500" : "bg-amber-500"
                              )}
                              style={{ width: `${emp.risk_score}%` }}
                            />
                          </div>
                          <span className="text-xs font-bold font-mono">{emp.risk_score}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button className="p-2 hover:bg-indigo-50 text-indigo-600 rounded-lg transition-colors">
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="p-4 bg-slate-50 border-t border-slate-200 text-center">
              <button className="text-xs font-bold text-indigo-600 hover:text-indigo-700 uppercase tracking-widest">
                Xem tất cả danh sách rủi ro
              </button>
            </div>
          </div>
        </div>

        {/* Data Reference Section */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-indigo-900 text-white p-8 rounded-2xl shadow-xl relative overflow-hidden">
            <div className="relative z-10">
              <h2 className="text-2xl font-bold mb-4">Mô hình Machine Learning</h2>
              <p className="text-indigo-100 text-sm mb-6 leading-relaxed">
                Hệ thống sử dụng thuật toán LightGBM với cơ chế Monthly Snapshot để dự báo xác suất nghỉ việc trong 3 tháng tới. 
                Dữ liệu được tổng hợp từ 4 nhóm chính: Nhân khẩu, Công ty, Nhân sự và Hành vi App.
              </p>
              <div className="flex flex-wrap gap-3">
                <span className="bg-white/10 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider">Feature Engineering</span>
                <span className="bg-white/10 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider">Time-based Split</span>
                <span className="bg-white/10 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider">Risk Tiering</span>
              </div>
            </div>
            <div className="absolute -right-12 -bottom-12 opacity-10">
              <RefreshCw className="w-64 h-64" />
            </div>
          </div>

          <div className="bg-slate-800 text-white p-8 rounded-2xl shadow-xl">
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
              <Search className="w-5 h-5 text-indigo-400" />
              Tín hiệu rủi ro (Early Warning)
            </h2>
            <div className="space-y-4">
              {[
                { label: 'Biến động chấm công', desc: 'Số ngày nghỉ không lương tăng > 50% so với TB 3 tháng.', icon: Clock },
                { label: 'Hành vi App', desc: 'Tần suất rút lương (EWA) tăng đột ngột hoặc ngừng truy cập App.', icon: Users },
                { label: 'Thay đổi thâm niên', desc: 'Nhân sự trong giai đoạn 3-6 tháng đầu có rủi ro cao nhất.', icon: Briefcase },
                { label: 'Vị trí địa lý', desc: 'IP truy cập thay đổi sang tỉnh thành khác nơi làm việc.', icon: MapPin },
              ].map((item, i) => (
                <div key={i} className="flex gap-4 items-start p-3 rounded-xl hover:bg-white/5 transition-colors">
                  <div className="p-2 bg-indigo-500/20 rounded-lg">
                    <item.icon className="w-4 h-4 text-indigo-400" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold">{item.label}</h4>
                    <p className="text-xs text-slate-400 mt-1">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm text-slate-500">© 2026 The Rolling Potato - Hệ thống dự báo Attrition v0.1 <span style={{ fontStyle: 'italic' }}>(for demonstration purpose only)</span></p>
        </div>
      </footer>
    </div>
  );
}
