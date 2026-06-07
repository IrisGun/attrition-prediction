import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import fs from "fs";
import { exec } from "child_process";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API routes FIRST
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok" });
  });

  // Serve ML artifacts
  app.get("/api/ml/predictions", (req, res) => {
    const predictionsPath = path.join(process.cwd(), 'ml/artifacts/predictions.json');
    if (fs.existsSync(predictionsPath)) {
      const data = fs.readFileSync(predictionsPath, 'utf8');
      res.json(JSON.parse(data));
    } else {
      // Return mock data if not found
      res.json({
        total_employees: 1240,
        high_risk_count: 156,
        avg_risk_score: 42,
        risk_distribution: [
          { name: 'High Risk', value: 156, color: '#ef4444' },
          { name: 'Medium Risk', value: 384, color: '#f59e0b' },
          { name: 'Low Risk', value: 700, color: '#10b981' },
        ],
        top_high_risk: [
          { emp_id: 'EMP_0042', company: 'TechCorp', dept: 'Production', tenure_months: 3, risk_score: 92 },
          { emp_id: 'EMP_0156', company: 'RetailCo', dept: 'Sales', tenure_months: 24, risk_score: 88 },
          { emp_id: 'EMP_0892', company: 'FactoryX', dept: 'Production', tenure_months: 6, risk_score: 85 },
          { emp_id: 'EMP_1024', company: 'SalesForce', dept: 'HR', tenure_months: 12, risk_score: 82 },
          { emp_id: 'EMP_0567', company: 'TechCorp', dept: 'IT', tenure_months: 4, risk_score: 79 },
        ]
      });
    }
  });

  // Trigger ML pipeline
  app.post("/api/ml/run", (req, res) => {
    exec('python3 ml/run_pipeline.py', (error, stdout, stderr) => {
      if (error) {
        console.error(`Error running pipeline: ${error.message}`);
        return res.status(500).json({ error: error.message, stderr });
      }
      console.log(`Pipeline output: ${stdout}`);
      res.json({ message: "ML Pipeline triggered successfully", stdout });
    });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
