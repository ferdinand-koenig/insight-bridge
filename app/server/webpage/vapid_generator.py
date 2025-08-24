import base64
import ecdsa

def generate_vapid_keypair():
  """
  Generate a new set of encoded key-pair for VAPID
  """
  pk = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p)
  vk = pk.get_verifying_key()

  return {
    'private_key': (base64.urlsafe_b64encode(pk.to_string()).strip(b"=")).decode('utf-8'),
    'public_key': (base64.urlsafe_b64encode(b"\x04" + vk.to_string()).strip(b"=")).decode('utf-8')
  }

