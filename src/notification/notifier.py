"""专业版 - 信号推送通知器

支持 Webhook 和邮件两种推送方式：
  - Webhook: 发送 JSON 到指定 URL（可对接钉钉/飞书/企业微信/Slack）
  - 邮件: SMTP 发送预测信号摘要
"""

from __future__ import annotations

import json
import logging
import smtplib
import ssl
import urllib.parse
import urllib.request
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)


class SignalNotifier:
    """预测信号推送器"""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        notify_cfg = cfg.get("notification", {})

        self.webhook_url = notify_cfg.get("webhook_url") or ""
        self.webhook_type = notify_cfg.get("webhook_type", "generic")  # generic/dingtalk/feishu/wechat

        self.smtp_host = notify_cfg.get("smtp_host") or ""
        self.smtp_port = notify_cfg.get("smtp_port", 465)
        self.smtp_user = notify_cfg.get("smtp_user") or ""
        self.smtp_password = notify_cfg.get("smtp_password") or ""
        self.email_from = notify_cfg.get("email_from") or self.smtp_user
        self.email_to = notify_cfg.get("email_to", [])
        if isinstance(self.email_to, str):
            self.email_to = [self.email_to]

        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def notify(self, predictions: list[dict[str, Any]]) -> dict[str, bool]:
        """推送预测信号，返回各渠道发送结果"""
        results = {"webhook": False, "email": False}
        if not predictions:
            return results

        summary = self._format_summary(predictions)

        if self.webhook_url:
            results["webhook"] = self._send_webhook(predictions, summary)
        if self.email_to:
            results["email"] = self._send_email(summary)

        logger.info(f"信号推送完成: {results}")
        return results

    def _format_summary(self, predictions: list[dict]) -> str:
        """格式化预测摘要"""
        lines = [
            f"📊 TrendCast Pro 预测信号 - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"共 {len(predictions)} 个标的",
            "",
        ]
        for p in predictions:
            symbol = p.get("symbol", "?")
            preds = p.get("predictions", {})
            for h, pred in preds.items():
                if "error" in pred:
                    continue
                direction = pred.get("direction", "?")
                proba = pred.get("probability", 0)
                conf = pred.get("confidence", 0)
                icon = "📈" if direction == "看涨" else "📉"
                lines.append(f"{icon} {symbol} [{h}]: {direction} (概率={proba:.0%}, 置信度={conf:.0%})")
        return "\n".join(lines)

    def _send_webhook(self, predictions: list[dict], summary: str) -> bool:
        """发送 Webhook 通知"""
        try:
            if self.webhook_type == "feishu":
                payload = {"msg_type": "text", "content": {"text": summary}}
            elif self.webhook_type == "dingtalk":
                payload = {"msgtype": "text", "text": {"content": summary}}
            elif self.webhook_type == "wechat":
                payload = {"msgtype": "text", "text": {"content": summary}}
            else:
                payload = {
                    "event": "prediction_signal",
                    "timestamp": datetime.now().isoformat(),
                    "summary": summary,
                    "predictions": predictions,
                }

            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "TrendCastPro/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15, context=self._ssl_ctx) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Webhook 推送失败: {e}")
            return False

    def _send_email(self, summary: str) -> bool:
        """发送邮件通知"""
        if not self.smtp_host or not self.smtp_user:
            logger.warning("SMTP 配置不完整，跳过邮件推送")
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"TrendCast Pro 预测信号 - {datetime.now().strftime('%Y-%m-%d')}"
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_to)

            html = f"<pre style='font-size:14px;line-height:1.6'>{summary}</pre>"
            msg.attach(MIMEText(summary, "plain", "utf-8"))
            msg.attach(MIMEText(html, "html", "utf-8"))

            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=self._ssl_ctx, timeout=20) as server:
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.email_from, self.email_to, msg.as_string())
            return True
        except Exception as e:
            logger.error(f"邮件推送失败: {e}")
            return False
