/**
 * G_One_Sync AI — Mobile TypeScript Interfaces
 * ============================================
 * Strongly-typed domain models matching the mobile serving API schemas.
 */

export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';

export type RiskTrend = 'improving' | 'stable' | 'worsening';

export interface PatientVitalsSummary {
  heart_rate: number;
  respiratory_rate: number;
  spo2_pct: number;
  temperature_c: number;
  systolic_bp: number;
  diastolic_bp: number;
}

export interface PatientLabsSummary {
  wbc_count: number;
  lactate: number;
  creatinine: number;
}

export interface RiskFactor {
  feature_name: string;
  display_name: string;
  shap_value: number;
  current_value: number;
  direction: 'increasing' | 'decreasing';
}

export interface PatientRiskSummary {
  patient_id: number;
  risk_level: RiskLevel;
  risk_score: number;
  deterioration_probability: number;
  vitals: PatientVitalsSummary;
  labs: PatientLabsSummary;
  top_risk_factors: RiskFactor[];
  last_updated: string;
  alerts: string[];
}

export interface WatchlistEntry {
  patient_id: number;
  risk_level: RiskLevel;
  risk_score: number;
  deterioration_probability: number;
  primary_concern: string;
  last_updated: string;
  trend: RiskTrend;
}

export interface WatchlistResponse {
  physician_id: string;
  patients: WatchlistEntry[];
  total_patients: number;
  critical_count: number;
  high_count: number;
  last_refreshed: string;
}

export interface NotificationHistoryEntry {
  notification_id: string;
  patient_id?: number;
  risk_level?: RiskLevel;
  title: string;
  body: string;
  sent_at: string;
  read: boolean;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserSession {
  username: string;
  accessToken: string;
  refreshToken: string;
  isAuthenticated: boolean;
}
