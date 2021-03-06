pragma solidity ^0.4.13;

/**
 * @title ERC20Basic
 * @dev Simpler version of ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/179
 */
contract ERC20Basic {
  uint256 public totalSupply;
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}



/**
 * @title ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/20
 */
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) public view returns (uint256);
  function transferFrom(address from, address to, uint256 value) public returns (bool);
  function approve(address spender, uint256 value) public returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
}



/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address public owner;


  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);


  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  function Ownable() public {
    owner = msg.sender;
  }


  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }


  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }

}




/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    return c;
  }

  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }

  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}
contract XMB is ERC20,Ownable{
	using SafeMath for uint256;

	//the base info of the token
	string public constant name="XMB";
	string public constant symbol="XMB";
	string public constant version = "1.0";
	uint256 public constant decimals = 18;

    mapping(address => uint256) balances;
	mapping (address => mapping (address => uint256)) allowed;
	//?????????10???
	uint256 public constant MAX_SUPPLY=1000000000*10**decimals;
	//????????????3?????????????????????????????????
	uint256 public constant INIT_SUPPLY=300000000*10**decimals;

	//????????????????????????
	uint256 public stepOneRate;
	//????????????????????????
	uint256 public stepTwoRate;

	//????????????????????????
	uint256 public stepOneStartTime;
	//????????????????????????
	uint256 public stepOneEndTime;


	//????????????????????????
	uint256 public stepTwoStartTime;
	//????????????????????????
	uint256 public stepTwoEndTime;

	//??????????????????1
	uint256 public stepOneLockEndTime;

	//??????????????????2
	uint256 public stepTwoLockEndTime;

	//???????????????
	uint256 public airdropSupply;

	//??????
    struct epoch  {
        uint256 endTime;
        uint256 amount;
    }

	//???????????????????????????
	mapping(address=>epoch[]) public lockEpochsMap;


	function XMB(){
		airdropSupply = 0;
		//????????????5w???
		stepOneRate = 50000;
		//????????????2.5w???
		stepTwoRate = 25000;
		//20180214 00:00:00
		stepOneStartTime=1518537600;
		//20180220 00:00:00
		stepOneEndTime=1519056000;


		//20180220 00:00:00
		stepTwoStartTime=1519056000;
		//20180225 00:00:00
		stepTwoEndTime=1519488000;

		//20180501 00:00:00
		stepOneLockEndTime = 1525104000;

		//20180401 00:00:00
		stepTwoLockEndTime = 1522512000;

		totalSupply = INIT_SUPPLY;
		balances[msg.sender] = INIT_SUPPLY;
		Transfer(0x0, msg.sender, INIT_SUPPLY);
	}

	modifier totalSupplyNotReached(uint256 _ethContribution,uint rate){
		assert(totalSupply.add(_ethContribution.mul(rate)) <= MAX_SUPPLY);
		_;
	}


	//??????
    function airdrop(address [] _holders,uint256 paySize) external
    	onlyOwner 
	{
        uint256 count = _holders.length;
        assert(paySize.mul(count) <= balanceOf(msg.sender));
        for (uint256 i = 0; i < count; i++) {
            transfer(_holders [i], paySize);
			airdropSupply = airdropSupply.add(paySize);
        }
    }


	//?????????????????????????????????
	function () payable external
	{
			if(now > stepOneStartTime&&now<=stepOneEndTime){
				processFunding(msg.sender,msg.value,stepOneRate);
				//????????????
				uint256 stepOnelockAmount = msg.value.mul(stepOneRate);
				lockBalance(msg.sender,stepOnelockAmount,stepOneLockEndTime);
			}else if(now > stepTwoStartTime&&now<=stepTwoEndTime){
				processFunding(msg.sender,msg.value,stepTwoRate);
				//????????????
				uint256 stepTwolockAmount = msg.value.mul(stepTwoRate);
				lockBalance(msg.sender,stepTwolockAmount,stepTwoLockEndTime);				
			}else{
				revert();
			}
	}

	//owner???????????????????????????eth
	function etherProceeds() external
		onlyOwner

	{
		if(!msg.sender.send(this.balance)) revert();
	}

	//????????????
	function lockBalance(address user, uint256 amount,uint256 endTime) internal
	{
		 epoch[] storage epochs = lockEpochsMap[user];
		 epochs.push(epoch(endTime,amount));
	}

	function processFunding(address receiver,uint256 _value,uint256 fundingRate) internal
		totalSupplyNotReached(_value,fundingRate)

	{
		uint256 tokenAmount = _value.mul(fundingRate);
		totalSupply=totalSupply.add(tokenAmount);
		balances[receiver] += tokenAmount;  // safeAdd not needed; bad semantics to use here
		Transfer(0x0, receiver, tokenAmount);
	}


	function setStepOneRate (uint256 _rate)  external 
		onlyOwner
	{
		stepOneRate=_rate;
	}
	function setStepTwoRate (uint256 _rate)  external 
		onlyOwner
	{
		stepTwoRate=_rate;
	}	

	function setStepOneTime (uint256 _stepOneStartTime,uint256 _stepOneEndTime)  external 
		onlyOwner
	{
		stepOneStartTime=_stepOneStartTime;
		stepOneEndTime = _stepOneEndTime;
	}	

	function setStepTwoTime (uint256 _stepTwoStartTime,uint256 _stepTwoEndTime)  external 
		onlyOwner
	{
		stepTwoStartTime=_stepTwoStartTime;
		stepTwoEndTime = _stepTwoEndTime;
	}	

	function setStepOneLockEndTime (uint256 _stepOneLockEndTime) external
		onlyOwner
	{
		stepOneLockEndTime = _stepOneLockEndTime;
	}
	
	function setStepTwoLockEndTime (uint256 _stepTwoLockEndTime) external
		onlyOwner
	{
		stepTwoLockEndTime = _stepTwoLockEndTime;
	}

  //???????????????????????????????????????????????????????????????????????????
  	function transfer(address _to, uint256 _value) public  returns (bool)
 	{
		require(_to != address(0));
		//??????????????????
		epoch[] epochs = lockEpochsMap[msg.sender];
		uint256 needLockBalance = 0;
		for(uint256 i;i<epochs.length;i++)
		{
			//??????????????????????????????????????????,???????????????
			if( now < epochs[i].endTime )
			{
				needLockBalance=needLockBalance.add(epochs[i].amount);
			}
		}

		require(balances[msg.sender].sub(_value)>=needLockBalance);
		// SafeMath.sub will throw if there is not enough balance.
		balances[msg.sender] = balances[msg.sender].sub(_value);
		balances[_to] = balances[_to].add(_value);
		Transfer(msg.sender, _to, _value);
		return true;
  	}

  	function balanceOf(address _owner) public constant returns (uint256 balance) 
  	{
		return balances[_owner];
  	}


  //??????????????????????????????????????????????????????????????????-??????????????????????????????????????????
  	function transferFrom(address _from, address _to, uint256 _value) public returns (bool) 
  	{
		require(_to != address(0));

		//??????????????????
		epoch[] epochs = lockEpochsMap[_from];
		uint256 needLockBalance = 0;
		for(uint256 i;i<epochs.length;i++)
		{
			//??????????????????????????????????????????,???????????????
			if( now < epochs[i].endTime )
			{
				needLockBalance = needLockBalance.add(epochs[i].amount);
			}
		}

		require(balances[_from].sub(_value)>=needLockBalance);
		uint256 _allowance = allowed[_from][msg.sender];

		balances[_from] = balances[_from].sub(_value);
		balances[_to] = balances[_to].add(_value);
		allowed[_from][msg.sender] = _allowance.sub(_value);
		Transfer(_from, _to, _value);
		return true;
  	}

  	function approve(address _spender, uint256 _value) public returns (bool) 
  	{
		allowed[msg.sender][_spender] = _value;
		Approval(msg.sender, _spender, _value);
		return true;
  	}

  	function allowance(address _owner, address _spender) public constant returns (uint256 remaining) 
  	{
		return allowed[_owner][_spender];
  	}

	  
}