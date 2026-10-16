/**
 * G_One_Sync AI — Mobile Real-Time WebSocket Client
 * ====================================================
 * Connects to the streaming clinical deterioration feed (/ws/live)
 * to receive real-time vitals and prediction alerts.
 */

import { WatchlistEntry } from '../types';
import { WS_BASE_URL } from '../utils/constants';

type PatientUpdateListener = (patient: WatchlistEntry) => void;
type ConnectionStateListener = (connected: boolean) => void;

class WebSocketService {
  private socket: WebSocket | null = null;
  private url: string = WS_BASE_URL;
  private reconnectIntervalMs: number = 5000;
  private shouldReconnect: boolean = true;
  private updateListeners: Set<PatientUpdateListener> = new Set();
  private stateListeners: Set<ConnectionStateListener> = new Set();

  public setUrl(url: string) {
    this.url = url;
    if (this.socket) {
      this.disconnect();
      this.connect();
    }
  }

  public connect(): void {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.shouldReconnect = true;

    try {
      this.socket = new WebSocket(this.url);

      this.socket.onopen = () => {
        console.log('[Mobile WS] Connected to clinical stream:', this.url);
        this.notifyState(true);
      };

      this.socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          // Handle telemetry / patient updates
          if (payload && payload.patient_id) {
            this.notifyPatientUpdate(payload as WatchlistEntry);
          }
        } catch (e) {
          console.warn('[Mobile WS] Failed to parse message:', e);
        }
      };

      this.socket.onerror = (err) => {
        console.warn('[Mobile WS] WebSocket error:', err);
      };

      this.socket.onclose = () => {
        console.log('[Mobile WS] Disconnected');
        this.notifyState(false);
        this.socket = null;
        if (this.shouldReconnect) {
          setTimeout(() => this.connect(), this.reconnectIntervalMs);
        }
      };
    } catch (err) {
      console.warn('[Mobile WS] Connection attempt failed:', err);
      this.notifyState(false);
      if (this.shouldReconnect) {
        setTimeout(() => this.connect(), this.reconnectIntervalMs);
      }
    }
  }

  public disconnect(): void {
    this.shouldReconnect = false;
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.notifyState(false);
  }

  public onPatientUpdate(listener: PatientUpdateListener): () => void {
    this.updateListeners.add(listener);
    return () => this.updateListeners.delete(listener);
  }

  public onConnectionStateChange(listener: ConnectionStateListener): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  private notifyPatientUpdate(data: WatchlistEntry) {
    this.updateListeners.forEach((listener) => {
      try {
        listener(data);
      } catch (e) {
        console.error('Error in patient update listener:', e);
      }
    });
  }

  private notifyState(connected: boolean) {
    this.stateListeners.forEach((listener) => {
      try {
        listener(connected);
      } catch (e) {
        console.error('Error in WS state listener:', e);
      }
    });
  }
}

export const wsService = new WebSocketService();
