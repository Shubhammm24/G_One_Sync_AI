/**
 * G_One_Sync AI — SHAP Chart Component
 * =====================================
 * Visualizes SHAP feature attributions on mobile to explain
 * why a patient is deteriorating.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { RiskFactor } from '../types';
import { THEME } from '../utils/constants';

interface SHAPChartProps {
  factors: RiskFactor[];
}

export const SHAPChart: React.FC<SHAPChartProps> = ({ factors }) => {
  if (!factors || factors.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyText}>No prominent risk drivers identified.</Text>
      </View>
    );
  }

  // Normalize max absolute SHAP value for bar scaling
  const maxShap = Math.max(...factors.map((f) => Math.abs(f.shap_value)), 0.01);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Key Deterioration Factors (SHAP)</Text>
      <Text style={styles.subtitle}>
        Features driving predicted risk up or down
      </Text>

      {factors.map((factor, index) => {
        const isIncreasing = factor.direction === 'increasing' || factor.shap_value > 0;
        const barWidthPct = Math.min(Math.round((Math.abs(factor.shap_value) / maxShap) * 100), 100);
        const barColor = isIncreasing ? '#EF4444' : '#10B981';

        return (
          <View key={`${factor.feature_name}-${index}`} style={styles.factorRow}>
            <View style={styles.labelRow}>
              <Text style={styles.factorName}>{factor.display_name || factor.feature_name}</Text>
              <Text style={[styles.factorValue, { color: barColor }]}>
                {isIncreasing ? '+' : ''}
                {factor.shap_value.toFixed(2)} (Val: {factor.current_value})
              </Text>
            </View>

            <View style={styles.barBackground}>
              <View
                style={[
                  styles.barFill,
                  {
                    width: `${barWidthPct}%`,
                    backgroundColor: barColor,
                  },
                ]}
              />
            </View>
          </View>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: THEME.cardBg,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: THEME.cardBorder,
    marginTop: 12,
  },
  title: {
    color: THEME.textPrimary,
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 2,
  },
  subtitle: {
    color: THEME.textMuted,
    fontSize: 12,
    marginBottom: 14,
  },
  emptyContainer: {
    padding: 20,
    alignItems: 'center',
    backgroundColor: THEME.cardBg,
    borderRadius: 12,
  },
  emptyText: {
    color: THEME.textMuted,
    fontSize: 13,
  },
  factorRow: {
    marginBottom: 12,
  },
  labelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  factorName: {
    color: THEME.textSecondary,
    fontSize: 13,
    fontWeight: '600',
  },
  factorValue: {
    fontSize: 12,
    fontWeight: '700',
  },
  barBackground: {
    height: 6,
    backgroundColor: THEME.surface,
    borderRadius: 3,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 3,
  },
});
