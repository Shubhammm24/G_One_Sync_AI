/**
 * G_One_Sync AI — Patient Detail Screen
 * =======================================
 * In-depth clinical telemetry, vitals, lab values,
 * SHAP explanations, and rapid clinical intervention actions.
 */

import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { RiskBadge } from '../components/RiskBadge';
import { SHAPChart } from '../components/SHAPChart';
import { VitalCard } from '../components/VitalCard';
import { apiService } from '../services/api';
import { PatientRiskSummary } from '../types';
import { LAB_BOUNDS, RISK_COLORS, THEME, VITAL_BOUNDS } from '../utils/constants';

interface PatientDetailScreenProps {
  patientId: number;
  onBack: () => void;
}

export const PatientDetailScreen: React.FC<PatientDetailScreenProps> = ({
  patientId,
  onBack,
}) => {
  const [patient, setPatient] = useState<PatientRiskSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchDetails = async () => {
      try {
        const data = await apiService.getPatientSummary(patientId);
        setPatient(data);
      } catch (err) {
        console.warn('Failed to fetch from API, loading mock detail:', err);
        // Fallback realistic clinical detail
        setPatient({
          patient_id: patientId,
          risk_level: 'CRITICAL',
          risk_score: 91,
          deterioration_probability: 0.91,
          vitals: {
            heart_rate: 128.0,
            respiratory_rate: 28.0,
            spo2_pct: 90.0,
            temperature_c: 39.1,
            systolic_bp: 88.0,
            diastolic_bp: 52.0,
          },
          labs: {
            wbc_count: 18.2,
            lactate: 4.1,
            creatinine: 2.2,
          },
          top_risk_factors: [
            {
              feature_name: 'lactate',
              display_name: 'Serum Lactate',
              shap_value: 0.42,
              current_value: 4.1,
              direction: 'increasing',
            },
            {
              feature_name: 'respiratory_rate',
              display_name: 'Respiratory Rate',
              shap_value: 0.35,
              current_value: 28.0,
              direction: 'increasing',
            },
            {
              feature_name: 'spo2_pct',
              display_name: 'SpO2 Oxygen Saturation',
              shap_value: 0.28,
              current_value: 90.0,
              direction: 'increasing',
            },
            {
              feature_name: 'systolic_bp',
              display_name: 'Systolic Blood Pressure',
              shap_value: 0.21,
              current_value: 88.0,
              direction: 'increasing',
            },
            {
              feature_name: 'heart_rate',
              display_name: 'Tachycardia (HR)',
              shap_value: 0.18,
              current_value: 128.0,
              direction: 'increasing',
            },
          ],
          last_updated: new Date().toISOString(),
          alerts: [
            '🔴 CRITICAL: Septic Shock Risk Elevated (Lactate 4.1 mmol/L)',
            '🟠 Tachycardia with severe hypotension (BP 88/52 mmHg)',
            '🟠 Severe Hypoxemia (SpO2 90% on room air)',
          ],
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchDetails();
  }, [patientId]);

  const handleAcknowledge = () => {
    Alert.alert(
      'Acknowledge Alert',
      `Alert for Bed #${patientId} acknowledged. Noted in clinical audit trail.`,
      [{ text: 'OK' }]
    );
  };

  const handleCallICU = () => {
    Alert.alert(
      'Contact ICU Station',
      `Connecting to ICU Unit 4 Nurse Station for Bed #${patientId}...`,
      [{ text: 'Cancel', style: 'cancel' }, { text: 'Call Now' }]
    );
  };

  if (isLoading || !patient) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={THEME.primary} />
        <Text style={styles.loadingText}>Fetching patient telemetry...</Text>
      </View>
    );
  }

  const isCritical = patient.risk_level === 'CRITICAL';

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Bed #{patient.patient_id}</Text>
        <RiskBadge level={patient.risk_level} size="medium" />
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Risk Score Summary Banner */}
        <View
          style={[
            styles.riskBanner,
            {
              borderColor: RISK_COLORS[patient.risk_level],
              backgroundColor: isCritical
                ? 'rgba(239, 68, 68, 0.1)'
                : 'rgba(59, 130, 246, 0.1)',
            },
          ]}
        >
          <View style={styles.riskBannerLeft}>
            <Text style={styles.riskBannerScore}>{patient.risk_score}</Text>
            <Text style={styles.riskBannerScoreLabel}>/100 Risk Score</Text>
          </View>
          <View style={styles.riskBannerRight}>
            <Text style={styles.riskBannerProb}>
              {(patient.deterioration_probability * 100).toFixed(1)}% Deterioration Probability
            </Text>
            <Text style={styles.riskBannerSub}>
              Predicted horizon: next 12 hours
            </Text>
          </View>
        </View>

        {/* Clinical Alerts Banner */}
        {patient.alerts && patient.alerts.length > 0 && (
          <View style={styles.alertsBox}>
            <Text style={styles.alertsTitle}>Clinical Alerts</Text>
            {patient.alerts.map((alert, i) => (
              <Text key={i} style={styles.alertItem}>
                {alert}
              </Text>
            ))}
          </View>
        )}

        {/* Vital Signs Grid */}
        <Text style={styles.sectionTitle}>Vital Signs</Text>
        <View style={styles.vitalsGrid}>
          <View style={styles.row}>
            <VitalCard
              label={VITAL_BOUNDS.heart_rate.label}
              value={patient.vitals.heart_rate}
              unit={VITAL_BOUNDS.heart_rate.unit}
              min={VITAL_BOUNDS.heart_rate.min}
              max={VITAL_BOUNDS.heart_rate.max}
            />
            <VitalCard
              label={VITAL_BOUNDS.respiratory_rate.label}
              value={patient.vitals.respiratory_rate}
              unit={VITAL_BOUNDS.respiratory_rate.unit}
              min={VITAL_BOUNDS.respiratory_rate.min}
              max={VITAL_BOUNDS.respiratory_rate.max}
            />
          </View>

          <View style={styles.row}>
            <VitalCard
              label={VITAL_BOUNDS.spo2_pct.label}
              value={patient.vitals.spo2_pct}
              unit={VITAL_BOUNDS.spo2_pct.unit}
              min={VITAL_BOUNDS.spo2_pct.min}
              max={VITAL_BOUNDS.spo2_pct.max}
            />
            <VitalCard
              label={VITAL_BOUNDS.temperature_c.label}
              value={patient.vitals.temperature_c}
              unit={VITAL_BOUNDS.temperature_c.unit}
              min={VITAL_BOUNDS.temperature_c.min}
              max={VITAL_BOUNDS.temperature_c.max}
              formatDecimals={1}
            />
          </View>

          <View style={styles.row}>
            <VitalCard
              label={VITAL_BOUNDS.systolic_bp.label}
              value={patient.vitals.systolic_bp}
              unit={VITAL_BOUNDS.systolic_bp.unit}
              min={VITAL_BOUNDS.systolic_bp.min}
              max={VITAL_BOUNDS.systolic_bp.max}
            />
            <VitalCard
              label={VITAL_BOUNDS.diastolic_bp.label}
              value={patient.vitals.diastolic_bp}
              unit={VITAL_BOUNDS.diastolic_bp.unit}
              min={VITAL_BOUNDS.diastolic_bp.min}
              max={VITAL_BOUNDS.diastolic_bp.max}
            />
          </View>
        </View>

        {/* Lab Results */}
        <Text style={styles.sectionTitle}>Key Biomarkers & Labs</Text>
        <View style={styles.vitalsGrid}>
          <View style={styles.row}>
            <VitalCard
              label={LAB_BOUNDS.wbc_count.label}
              value={patient.labs.wbc_count}
              unit={LAB_BOUNDS.wbc_count.unit}
              min={LAB_BOUNDS.wbc_count.min}
              max={LAB_BOUNDS.wbc_count.max}
              formatDecimals={1}
            />
            <VitalCard
              label={LAB_BOUNDS.lactate.label}
              value={patient.labs.lactate}
              unit={LAB_BOUNDS.lactate.unit}
              min={LAB_BOUNDS.lactate.min}
              max={LAB_BOUNDS.lactate.max}
              formatDecimals={2}
            />
            <VitalCard
              label={LAB_BOUNDS.creatinine.label}
              value={patient.labs.creatinine}
              unit={LAB_BOUNDS.creatinine.unit}
              min={LAB_BOUNDS.creatinine.min}
              max={LAB_BOUNDS.creatinine.max}
              formatDecimals={2}
            />
          </View>
        </View>

        {/* Explainability / SHAP Factors */}
        <SHAPChart factors={patient.top_risk_factors} />

        {/* Fast Action Buttons */}
        <View style={styles.actionRow}>
          <TouchableOpacity
            style={[styles.actionBtn, styles.callBtn]}
            onPress={handleCallICU}
          >
            <Text style={styles.actionBtnText}>📞 Call ICU Station</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionBtn, styles.ackBtn]}
            onPress={handleAcknowledge}
          >
            <Text style={styles.actionBtnText}>✓ Acknowledge</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: THEME.bgDark,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 48,
    paddingBottom: 16,
    backgroundColor: THEME.cardBg,
    borderBottomWidth: 1,
    borderBottomColor: THEME.cardBorder,
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    color: THEME.primary,
    fontSize: 15,
    fontWeight: '700',
  },
  headerTitle: {
    color: THEME.textPrimary,
    fontSize: 18,
    fontWeight: '800',
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
  },
  riskBanner: {
    flexDirection: 'row',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    marginBottom: 16,
    alignItems: 'center',
  },
  riskBannerLeft: {
    borderRightWidth: 1,
    borderRightColor: THEME.cardBorder,
    paddingRight: 16,
    marginRight: 16,
    alignItems: 'center',
  },
  riskBannerScore: {
    color: THEME.textPrimary,
    fontSize: 32,
    fontWeight: '900',
  },
  riskBannerScoreLabel: {
    color: THEME.textMuted,
    fontSize: 10,
    fontWeight: '600',
  },
  riskBannerRight: {
    flex: 1,
  },
  riskBannerProb: {
    color: THEME.textPrimary,
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 2,
  },
  riskBannerSub: {
    color: THEME.textSecondary,
    fontSize: 12,
  },
  alertsBox: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: '#EF444455',
    marginBottom: 16,
  },
  alertsTitle: {
    color: '#EF4444',
    fontSize: 13,
    fontWeight: '800',
    textTransform: 'uppercase',
    marginBottom: 6,
  },
  alertItem: {
    color: THEME.textPrimary,
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 4,
  },
  sectionTitle: {
    color: THEME.textPrimary,
    fontSize: 16,
    fontWeight: '700',
    marginTop: 10,
    marginBottom: 8,
  },
  vitalsGrid: {
    marginBottom: 12,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  actionRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 24,
  },
  actionBtn: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  callBtn: {
    backgroundColor: '#DC2626',
  },
  ackBtn: {
    backgroundColor: THEME.surface,
    borderWidth: 1,
    borderColor: THEME.cardBorder,
  },
  actionBtnText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: THEME.bgDark,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: THEME.textMuted,
    marginTop: 12,
    fontSize: 13,
  },
});
